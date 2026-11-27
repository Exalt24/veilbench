import pytest

from veilbench import ids
from veilbench.corpus import people as P
from veilbench.detect.base import merge_spans
from veilbench.detect.rules import RuleDetector
from veilbench.schema import Span

R = RuleDetector()


def labels(text: str) -> list[tuple[str, str]]:
    return [(s.label, text[s.start:s.end]) for s in R.detect(text)]


def test_a_luhn_valid_card_is_found_and_a_same_length_order_number_is_not():
    assert ("CREDIT_CARD", "4111 1111 1111 1111") in labels("Paid with 4111 1111 1111 1111 today")
    assert labels("Tracking number 4111111111111112 shipped") == []


def test_iban_needs_a_valid_check_digit():
    assert ("IBAN", "GB82 WEST 1234 5698 7654 32") in labels("Refund to GB82 WEST 1234 5698 7654 32 please")
    assert not [x for x in labels("Refund to GB82 WEST 1234 5698 7654 33 please") if x[0] == "IBAN"]


def test_national_ids_by_checksum_and_by_shape():
    assert ("NATIONAL_ID", "12345678Z") in labels("DNI 12345678Z")
    assert not [x for x in labels("code 12345678A") if x[0] == "NATIONAL_ID"]
    assert ("NATIONAL_ID", "S1234567D") in labels("NRIC S1234567D")
    assert ("NATIONAL_ID", "123-45-6789") in labels("SSN 123-45-6789")
    assert ("NATIONAL_ID", "34-1234567-8") in labels("SSS 34-1234567-8")
    assert not [x for x in labels("ref 000-12-3456") if x[0] == "NATIONAL_ID"]


@pytest.mark.parametrize("negative", [
    "ORD-482913", "v2.14.3", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b", "81584d8c-4fa2-415d-a802-827283e0ad84",
    "SKU-4417-22", "1725643433", "build 10.2.3.4 shipped", "firmware is 6.4.17.140 on the unit", "Total 1234.56",
])
def test_things_that_look_like_identifiers_are_left_alone(negative):
    assert labels(negative) == [], negative


def test_obfuscated_forms_are_found():
    assert ("EMAIL", "john dot smith at example dot com") in labels("write to john dot smith at example dot com now")
    assert ("EMAIL", "ana[at]example[dot]org") in labels("mail ana[at]example[dot]org")
    assert ("PHONE", "9 1 7 5 5 5 0 1 4 2") in labels("call 9 1 7 5 5 5 0 1 4 2")
    words = "nine one seven five five five zero one four two"
    assert ("PHONE", words) in labels(f"it is {words}")
    assert ("NATIONAL_ID", "1 2 3 4 5 6 7 8 Z") in labels("My DNI is 1 2 3 4 5 6 7 8 Z.")
    assert ("EMAIL", "ting.lu%40corp.example") in labels("GET /users/ting.lu%40corp.example?x=1")


def test_the_words_form_prefers_the_longest_real_address():
    got = labels("Fatima at schmid dot sebastian at post dot example dot net")
    assert ("EMAIL", "schmid dot sebastian at post dot example dot net") in got


def test_secrets_in_several_vendor_shapes_and_by_context():
    for kind in ["stripe", "aws", "github", "slack", "google", "openai", "gitlab", "sendgrid", "hf", "jwt"]:
        label, value = P.make_secret(__import__("random").Random(1), kind)
        found = labels(f"credential: {value} end")
        assert (label, value) in found, kind
    assert ("PASSWORD", "Tiger#2024ban") in labels("the password is Tiger#2024ban")
    assert ("PASSWORD", "Tiger#2024ban") in labels("body: email=a@b.example&password=Tiger#2024ban&x=1")
    assert [x for x in labels('config token: "ab12cd34ef56ab12cd34ef56ab"') if x[0] == "API_KEY"]
    # a bare token with no context is only called a key when it mixes cases and digits; an all lowercase one is read as a hash or an id (a documented limit)
    assert [x for x in labels("seen A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6 today") if x[0] == "API_KEY"]
    assert labels("seen q7w3e9r2t5y8u1i4o6p0a3s5d7f9g2h4 today") == []


def test_a_date_is_a_birth_date_only_with_a_birth_cue():
    assert ("DATE_OF_BIRTH", "1984-03-07") in labels("Date of birth: 1984-03-07")
    assert ("DATE_OF_BIRTH", "March 7, 1984") in labels("I was born on March 7, 1984.")
    assert ("DATE_OF_BIRTH", "07/03/1984") in labels("confirm your date of birth?\nCaller: 07/03/1984.")
    assert labels("Order placed on 2026-03-08") == []


def test_names_come_from_cues_not_from_a_list():
    assert ("PERSON", "Maria Santos") in labels("Hello, my name is Maria Santos. Thanks.")
    assert ("PERSON", "Ms. Reyes") in labels("Please ask Ms. Reyes about it")
    assert labels("Hello team, the package is late") == []
    assert labels("Hi There, the parcel is late") == []                # a capitalised greeting word after a cue is not a name
    assert labels("Dear Support, the order is late") == []
    got = labels("account mgarcia@example.com ... later Maria Garcia called")
    assert not any(x[0] == "PERSON" for x in got)  # one part of an e-mail name is not enough
    got = labels("send to maria.garcia@example.com. Regards to Maria Garcia and Garcia, Maria.")
    assert [x for x in got if x[0] == "PERSON"] == [("PERSON", "Maria Garcia"), ("PERSON", "Garcia, Maria")]


def test_addresses_in_three_layouts():
    assert ("ADDRESS", "1450 Market St, San Francisco, CA 94103") in labels("ship to 1450 Market St, San Francisco, CA 94103 please")
    assert [x for x in labels("27 Mabini St., Cebu City 6000 is the place") if x[0] == "ADDRESS"]
    assert [x for x in labels("Calle de Alcala 42, 28014 Madrid") if x[0] == "ADDRESS"]


def test_rules_never_return_overlapping_spans(corpus):
    for rec in corpus:
        spans = sorted(R.detect(rec.text))
        for a, b in zip(spans, spans[1:]):
            assert a.end <= b.start, (rec.id, rec.text[a.start:b.end])


def test_every_validated_identifier_in_the_corpus_is_recovered(corpus):
    """Cards, IBANs, keys, passwords and addresses of the plain forms are exact: the rules must find all of them."""
    missed = []
    for rec in corpus:
        found = R.detect(rec.text)
        for g in rec.spans:
            if g.label in ("CREDIT_CARD", "IBAN", "JWT", "PASSWORD", "IP_ADDRESS") and "obfuscated" not in rec.slices:
                if not any(f.label == g.label and g.iou(f) >= 0.5 for f in found):
                    missed.append((rec.id, g.label, rec.text[g.start:g.end]))
    assert not missed, missed[:5]


# ---- merge_spans


def S(a, b, label, source):
    return Span(a, b, label, source)


def test_merge_keeps_the_better_detector_boundaries_and_drops_a_sloppy_copy():
    out = merge_spans([S(10, 20, "EMAIL", "rules"), S(8, 21, "PERSON", "stat")])
    assert [(s.start, s.end, s.label) for s in out] == [(10, 20, "EMAIL")]


def test_merge_keeps_a_finding_only_a_weaker_detector_made_and_trims_the_overlap():
    out = merge_spans([S(0, 10, "PERSON", "rules"), S(5, 30, "ADDRESS", "gliner")])
    assert [(s.start, s.end, s.label) for s in out] == [(0, 10, "PERSON"), (10, 30, "ADDRESS")]


def test_merge_label_follows_priority_and_output_never_overlaps():
    out = merge_spans([S(0, 5, "PHONE", "stat"), S(0, 5, "NATIONAL_ID", "rules"), S(3, 9, "PERSON", "spacy")])
    assert out[0].label == "NATIONAL_ID"
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start


def test_merge_covers_everything_any_detector_flagged_that_a_better_one_did_not():
    spans = [S(0, 4, "PERSON", "rules"), S(20, 30, "PERSON", "stat"), S(40, 41, "PERSON", "spacy")]
    covered = {i for s in merge_spans(spans) for i in range(s.start, s.end)}
    assert set(range(0, 4)) | set(range(20, 30)) | {40} == covered


def test_ids_module_is_what_the_rules_use():
    assert ids.luhn_valid("4111111111111111")
