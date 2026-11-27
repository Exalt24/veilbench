"""Format preserving encryption of digit strings: NIST SP 800-38G FF1 over AES, radix 10.

Checked against the published NIST sample vectors in tests/test_fpe.py. Used to mask structured identifiers so the output
still has the same length and separators (a phone number stays phone shaped, a card number stays Luhn valid).
"""
from __future__ import annotations

import math

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from veilbench import ids

RADIX = 10


def _aes_block(key: bytes, block: bytes) -> bytes:
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()  # a single block: ECB is the raw block cipher FF1 specifies
    return enc.update(block) + enc.finalize()


def _prf(key: bytes, data: bytes) -> bytes:
    y = bytes(16)
    for i in range(0, len(data), 16):
        y = _aes_block(key, bytes(a ^ b for a, b in zip(y, data[i:i + 16])))
    return y


def _num(digits: list[int]) -> int:
    n = 0
    for d in digits:
        n = n * RADIX + d
    return n


def _digits(n: int, length: int) -> list[int]:
    out = [0] * length
    for i in range(length - 1, -1, -1):
        out[i] = n % RADIX
        n //= RADIX
    return out


def _check(key: bytes, digits: list[int]) -> None:
    if len(key) not in (16, 24, 32):
        raise ValueError("the key must be 16, 24 or 32 bytes")
    if len(digits) < 6:
        raise ValueError("FF1 needs radix**n >= 1,000,000, so at least 6 digits")
    if any(not 0 <= d < RADIX for d in digits):
        raise ValueError("digits only")


def _round_inputs(key: bytes, tweak: bytes, n: int, u: int, v: int, b: int, d: int):
    p = bytes([1, 2, 1]) + RADIX.to_bytes(3, "big") + bytes([10, u % 256]) + n.to_bytes(4, "big") + len(tweak).to_bytes(4, "big")

    def y_for(i: int, other: list[int]) -> int:
        q = tweak + bytes((-len(tweak) - b - 1) % 16) + bytes([i]) + _num(other).to_bytes(b, "big")
        r = _prf(key, p + q)
        s = r
        for j in range(1, math.ceil(d / 16)):
            s += _aes_block(key, bytes(a ^ b_ for a, b_ in zip(r, j.to_bytes(16, "big"))))
        return int.from_bytes(s[:d], "big")

    return y_for


def ff1_encrypt(key: bytes, tweak: bytes, digits: list[int]) -> list[int]:
    _check(key, digits)
    n = len(digits)
    u, v = n // 2, n - n // 2
    b = math.ceil(math.ceil(v * math.log2(RADIX)) / 8)
    d = 4 * math.ceil(b / 4) + 4
    y_for = _round_inputs(key, tweak, n, u, v, b, d)
    a, bb = digits[:u], digits[u:]
    for i in range(10):
        m = u if i % 2 == 0 else v
        c = (_num(a) + y_for(i, bb)) % (RADIX ** m)
        a, bb = bb, _digits(c, m)
    return a + bb


def ff1_decrypt(key: bytes, tweak: bytes, digits: list[int]) -> list[int]:
    _check(key, digits)
    n = len(digits)
    u, v = n // 2, n - n // 2
    b = math.ceil(math.ceil(v * math.log2(RADIX)) / 8)
    d = 4 * math.ceil(b / 4) + 4
    y_for = _round_inputs(key, tweak, n, u, v, b, d)
    a, bb = digits[:u], digits[u:]
    for i in range(9, -1, -1):
        m = u if i % 2 == 0 else v
        c = (_num(bb) - y_for(i, a)) % (RADIX ** m)
        a, bb = _digits(c, m), a
    return a + bb


def mask_digits(key: bytes, tweak: bytes, text: str, keep_last: int = 0) -> str:
    """Encrypt the digits of `text` in place; every non digit character stays where it is. Short digit runs are returned unchanged."""
    positions = [i for i, ch in enumerate(text) if ch.isdigit()]
    if len(positions) <= keep_last or len(positions) - keep_last < 6:
        return text
    body = positions[:len(positions) - keep_last] if keep_last else positions
    enc = ff1_encrypt(key, tweak, [int(text[i]) for i in body])
    out = list(text)
    for i, dgt in zip(body, enc):
        out[i] = str(dgt)
    return "".join(out)


def mask_card(key: bytes, text: str) -> str:
    """Card number: keep the first digit (the network), encrypt the rest but the last, and recompute the Luhn digit so it stays valid."""
    pos = [i for i, ch in enumerate(text) if ch.isdigit()]
    if len(pos) < 13:
        return mask_digits(key, b"card", text)
    mid = pos[1:-1]
    enc = ff1_encrypt(key, b"card", [int(text[i]) for i in mid])
    out = list(text)
    for i, dgt in zip(mid, enc):
        out[i] = str(dgt)
    digits = "".join(out[i] for i in pos[:-1])
    fixed = ids.luhn_complete(digits, len(pos))
    out[pos[-1]] = fixed[-1]
    return "".join(out)


def mask_iban(key: bytes, text: str) -> str:
    """IBAN: encrypt the digits after the check digits, then recompute the check digits so the IBAN still validates."""
    compact = "".join(text.split()).upper()
    if not ids.iban_valid(compact):
        return mask_digits(key, b"iban", text)
    country, bban = compact[:2], compact[4:]
    digit_pos = [i for i, ch in enumerate(bban) if ch.isdigit()]
    if len(digit_pos) >= 6:
        enc = ff1_encrypt(key, b"iban", [int(bban[i]) for i in digit_pos])
        chars = list(bban)
        for i, dgt in zip(digit_pos, enc):
            chars[i] = str(dgt)
        bban = "".join(chars)
    new = ids.iban_make(country, bban)
    if " " in text:
        return " ".join(new[i:i + 4] for i in range(0, len(new), 4))
    return new


def mask_spanish_dni(key: bytes, text: str) -> str | None:
    m = [i for i, ch in enumerate(text) if ch.isdigit()]
    if len(m) == 8 and text[-1].isalpha():
        enc = ff1_encrypt(key, b"dni", [int(text[i]) for i in m])
        out = list(text)
        for i, dgt in zip(m, enc):
            out[i] = str(dgt)
        out[-1] = ids.dni_letter(int("".join(str(x) for x in enc)))
        return "".join(out)
    return None
