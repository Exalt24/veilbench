import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from veilbench import ids
from veilbench.transform import fpe

KEY = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")


def digits(s: str) -> list[int]:
    return [int(c) for c in s]


def test_ff1_matches_the_nist_sample_vectors():
    # NIST SP 800-38G sample data: key, tweak, plaintext -> ciphertext, radix 10
    assert "".join(map(str, fpe.ff1_encrypt(KEY, b"", digits("0123456789")))) == "2433477484"
    assert "".join(map(str, fpe.ff1_encrypt(KEY, bytes.fromhex("39383736353433323130"), digits("0123456789")))) == "6124200773"
    k192 = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F")
    assert "".join(map(str, fpe.ff1_encrypt(k192, b"", digits("0123456789")))) == "2830668132"


@settings(max_examples=60, deadline=None)
@given(st.text(alphabet="0123456789", min_size=6, max_size=24), st.binary(max_size=12))
def test_ff1_round_trip_and_alphabet(plain, tweak):
    enc = fpe.ff1_encrypt(KEY, tweak, digits(plain))
    assert len(enc) == len(plain) and all(0 <= d <= 9 for d in enc)
    assert fpe.ff1_decrypt(KEY, tweak, enc) == digits(plain)


def test_ff1_refuses_short_input_and_bad_keys():
    with pytest.raises(ValueError):
        fpe.ff1_encrypt(KEY, b"", digits("12345"))
    with pytest.raises(ValueError):
        fpe.ff1_encrypt(b"short", b"", digits("123456"))


@settings(max_examples=40, deadline=None)
@given(st.text(alphabet="0123456789", min_size=10, max_size=10))
def test_digit_masking_keeps_every_separator_in_place(d):
    shaped = f"+63 {d[:3]}-{d[3:6]} {d[6:]}"
    out = fpe.mask_digits(KEY, b"phone", shaped)
    assert len(out) == len(shaped)
    assert [c.isdigit() for c in out] == [c.isdigit() for c in shaped]
    assert re.sub(r"\d", "#", out) == re.sub(r"\d", "#", shaped)


@settings(max_examples=40, deadline=None)
@given(st.text(alphabet="0123456789", min_size=14, max_size=18))
def test_masked_card_is_still_luhn_valid_and_changes_the_number(body):
    card = ids.luhn_complete(body, len(body) + 1)
    assert ids.luhn_valid(card)
    out = fpe.mask_card(KEY, card)
    assert ids.luhn_valid(out) and len(out) == len(card) and out[0] == card[0]
    assert out != card or len(set(card[1:-1])) == 1  # a fixed point is possible in principle, never in practice


def test_masked_iban_still_validates():
    iban = ids.iban_make("ES", "21000418450200051332")
    assert ids.iban_valid(iban)
    out = fpe.mask_iban(KEY, iban)
    assert ids.iban_valid(out) and out != iban and out.startswith("ES")


def test_masked_dni_keeps_a_correct_check_letter():
    out = fpe.mask_spanish_dni(KEY, "12345678Z")
    assert out and ids.dni_valid(out) and out != "12345678Z"


def test_validators_accept_known_good_and_reject_one_digit_changes():
    assert ids.luhn_valid("4111111111111111") and not ids.luhn_valid("4111111111111112")
    assert ids.iban_valid("GB82WEST12345698765432") and not ids.iban_valid("GB82WEST12345698765433")
    assert ids.dni_valid("12345678Z") and not ids.dni_valid("12345678A")
    assert ids.nric_valid("S1234567D") and not ids.nric_valid("S1234567E")
    assert not ids.ssn_valid("000", "12", "3456") and not ids.ssn_valid("666", "12", "3456") and not ids.ssn_valid("912", "12", "3456")
    assert ids.ssn_valid("123", "45", "6789")


@settings(max_examples=60, deadline=None)
@given(st.text(alphabet="0123456789", min_size=6, max_size=15), st.integers(min_value=13, max_value=19))
def test_luhn_complete_always_produces_a_valid_number(prefix, length):
    prefix = prefix[: length - 1]
    assert ids.luhn_valid(ids.luhn_complete(prefix, length))
