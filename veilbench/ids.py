"""Checksummed identifier helpers shared by the generator (which builds valid values) and the rules (which validate them)."""
from __future__ import annotations

import re

DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
NRIC_ST = "JZIHGFEDCBA"
NRIC_WEIGHTS = (2, 7, 6, 5, 4, 3, 2)


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_valid(number: str) -> bool:
    d = digits_only(number)
    if len(d) < 12:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def luhn_complete(prefix_digits: str, length: int) -> str:
    """Append digits to `prefix_digits` up to `length`, the last one being the Luhn check digit. Filler is deterministic from the prefix."""
    body = prefix_digits
    seed = sum(int(c) * (i + 3) for i, c in enumerate(prefix_digits))
    while len(body) < length - 1:
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        body += str(seed % 10)
    total = 0
    for i, ch in enumerate(reversed(body)):
        n = int(ch)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return body + str((10 - total % 10) % 10)


def _iban_numeric(s: str) -> int:
    return int("".join(str(int(c, 36)) for c in s))


def iban_valid(iban: str) -> bool:
    s = re.sub(r"\s", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    return _iban_numeric(s[4:] + s[:4]) % 97 == 1


def iban_make(country: str, bban: str) -> str:
    check = 98 - _iban_numeric(bban + country + "00") % 97
    return f"{country}{check:02d}{bban}"


def dni_letter(number: int) -> str:
    return DNI_LETTERS[number % 23]


def dni_valid(s: str) -> bool:
    m = re.fullmatch(r"(\d{8})-?([A-Za-z])", s.strip())
    return bool(m) and dni_letter(int(m.group(1))) == m.group(2).upper()


def nric_letter(prefix: str, digits7: str) -> str:
    total = sum(int(d) * w for d, w in zip(digits7, NRIC_WEIGHTS))
    if prefix.upper() == "T":
        total += 4
    return NRIC_ST[total % 11]


def nric_valid(s: str) -> bool:
    m = re.fullmatch(r"([STst])(\d{7})([A-Za-z])", s.strip())
    return bool(m) and nric_letter(m.group(1), m.group(2)) == m.group(3).upper()


def ssn_valid(area: str, group: str, serial: str) -> bool:
    """US SSN shape rules: area not 000, 666 or 9xx, group not 00, serial not 0000."""
    return area not in ("000", "666") and not area.startswith("9") and group != "00" and serial != "0000"
