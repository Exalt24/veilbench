import random
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from veilbench.corpus import people as P
from veilbench.corpus.generate import generate_corpus
from veilbench.evaluation.metrics import verbatim_leak
from veilbench.transform.ops import TECHNIQUES, Transformer, canon_email, mask_surface

IDENTIFIER_TYPES = ["EMAIL", "PHONE", "CREDIT_CARD", "NATIONAL_ID", "IBAN", "ADDRESS", "DATE_OF_BIRTH", "IP_ADDRESS"]


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_no_ground_truth_identifier_survives_any_technique(corpus, technique):
    """With perfect detection, the output never contains an identifier of the record it came from."""
    tr = Transformer(technique, seed=4)
    bad = [(r.id, [r.text[s.start:s.end] for s in verbatim_leak(r, tr.apply(r.text, r.spans))]) for r in corpus]
    assert not [b for b in bad if b[1]], [b for b in bad if b[1]][:3]


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 5000), st.sampled_from(TECHNIQUES))
def test_no_identifier_survives_over_random_worlds(seed, technique):
    tr = Transformer(technique, seed=seed)
    for rec in generate_corpus(15, seed):
        assert not verbatim_leak(rec, tr.apply(rec.text, rec.spans)), (technique, rec.id)


def test_text_outside_the_spans_is_untouched(corpus):
    rec = next(r for r in corpus if r.kind == "ticket")
    out = Transformer("redact").apply(rec.text, rec.spans)
    # remove the placeholders and the original spans: what is left of both must be the same text
    stripped_out = re.sub(r"\[[A-Z_]+\]", "\0", out)
    stripped_in = rec.text
    for s in sorted(rec.spans, reverse=True):
        stripped_in = stripped_in[:s.start] + "\0" + stripped_in[s.end:]
    assert stripped_out == stripped_in


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 10_000))
def test_masking_preserves_length_and_punctuation(seed):
    rng = random.Random(seed)
    p = P.make_person(rng, ["en", "es", "ph", "de"])
    for label, surface in [("PHONE", P.phone_surface(rng, p)[0]), ("EMAIL", p.email), ("CREDIT_CARD", P.card_surface(rng, p)[0]),
                           ("NATIONAL_ID", p.national_id), ("ADDRESS", p.address), ("PERSON", p.full), ("DATE_OF_BIRTH", P.dob_surface(rng, p)[0])]:
        out = mask_surface(surface, label)
        assert len(out) == len(surface)
        # punctuation stays exactly where it was, and a letter or digit is either kept or hidden, never changed into another one
        assert all(a == b for a, b in zip(surface, out) if not a.isalnum())
        assert all(b in (a, "*") for a, b in zip(surface, out) if a.isalnum())


def test_masking_keeps_only_the_last_four_digits_of_a_number():
    assert mask_surface("4111 1111 1111 1111", "CREDIT_CARD") == "**** **** **** 1111"
    assert mask_surface("+63 917 555 0142", "PHONE") == "+** *** *** 0142"
    assert mask_surface("123-45-6789", "NATIONAL_ID") == "***-**-****"  # an ID keeps nothing
    assert mask_surface("sk_live_abcdef", "API_KEY") == "*" * 14  # a secret keeps nothing, not even where its separators were


def test_pseudonyms_are_the_same_every_time_within_one_key_and_differ_across_keys(corpus):
    a1, a2, b = Transformer("pseudonymize", key=b"key-A" * 4), Transformer("pseudonymize", key=b"key-A" * 4), Transformer("pseudonymize", key=b"key-B" * 4)
    same, differ, total = 0, 0, 0
    for rec in corpus[:150]:
        for s in rec.spans:
            if s.label in ("EMAIL", "PHONE", "CREDIT_CARD", "ADDRESS", "IP_ADDRESS", "IBAN"):
                v = rec.text[s.start:s.end]
                x1, x2, y = a1.replacement(v, s.label), a2.replacement(v, s.label), b.replacement(v, s.label)
                assert x1 == x2, (s.label, v)
                total += 1
                differ += int(x1 != y)
    assert total > 200 and differ / total > 0.98  # a different key almost never reproduces a pseudonym


def test_one_value_has_one_pseudonym_however_it_is_written():
    t = Transformer("pseudonymize")
    plain = t.replacement("john.smith@example.com", "EMAIL")
    assert plain == t.replacement("john dot smith at example dot com", "EMAIL") == t.replacement("john.smith[at]example[dot]com", "EMAIL") == t.replacement("john.smith%40example.com", "EMAIL")
    assert canon_email("John.Smith (at) Example.com") == "john.smith@example.com"
    only_digits = lambda v: "".join(c for c in v if c.isdigit())
    assert only_digits(t.replacement("+63 917 555 0142", "PHONE"))[-8:] == only_digits(t.replacement("0917-555-0142", "PHONE"))[-8:]


def test_a_pseudonymized_join_key_still_joins_and_a_redacted_one_does_not():
    rows = ["mail maria@example.com", "again maria@example.com", "other pedro@example.com"]
    pseudo, redact = Transformer("pseudonymize"), Transformer("redact")
    from veilbench.detect.rules import RuleDetector
    det = RuleDetector()
    p_out = [pseudo.apply(r, det.detect(r)) for r in rows]
    r_out = [redact.apply(r, det.detect(r)) for r in rows]
    assert p_out[0].split()[1] == p_out[1].split()[1] != p_out[2].split()[1]
    assert len({o.split()[1] for o in r_out}) == 1


def test_secrets_are_removed_by_every_technique_never_pseudonymized():
    t = Transformer("pseudonymize")
    for label in ("API_KEY", "JWT", "PASSWORD"):
        assert t.replacement("abc123SECRETvalue", label) == f"[{label}]"
    s = Transformer("synthesize")
    assert s.replacement("hunter2hunter2", "PASSWORD") == "[PASSWORD]"


def test_pseudonymized_numbers_keep_their_shape_and_stay_valid():
    from veilbench import ids
    t = Transformer("pseudonymize")
    card = t.replacement("4111 1111 1111 1111", "CREDIT_CARD")
    assert ids.luhn_valid(card) and re.fullmatch(r"\d{4} \d{4} \d{4} \d{4}", card) and card != "4111 1111 1111 1111"
    iban = t.replacement("GB82 WEST 1234 5698 7654 32", "IBAN")
    assert ids.iban_valid(iban) and iban != "GB82 WEST 1234 5698 7654 32"
    dni = t.replacement("12345678Z", "NATIONAL_ID")
    assert ids.dni_valid(dni) and dni != "12345678Z"
    assert re.fullmatch(r"\+63 \d{3} \d{3} \d{4}", t.replacement("+63 917 555 0142", "PHONE"))
    d = t.replacement("1984-03-07", "DATE_OF_BIRTH")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) and d != "1984-03-07"


def test_pseudonym_names_are_never_real_names_from_the_pools():
    from veilbench.corpus import names as N
    from veilbench.transform.xdoc import fold
    real = {fold(t) for pool in N.POOLS.values() for k in ("first", "last") for t in pool[k]}
    t = Transformer("pseudonymize")
    for rec in generate_corpus(80, 9):
        for s in rec.spans:
            if s.label == "PERSON":
                for tok in re.findall(r"[A-Za-z]+", t.replacement(rec.text[s.start:s.end], "PERSON")):
                    assert fold(tok) not in real or tok.lower() in ("ms", "mr", "dr", "sr", "sra", "dra")


def test_anonymize_generalizes_instead_of_removing():
    a = Transformer("anonymize")
    assert a.replacement("1984-03-07", "DATE_OF_BIRTH") == "1980s"
    assert a.replacement("203.0.113.9", "IP_ADDRESS") == "203.0.0.0/16"
    assert a.replacement("maria@example.com", "EMAIL") == "[EMAIL]@example.com"
    assert a.replacement("+63 917 555 0142", "PHONE") == "+63 [PHONE]"
    assert a.replacement("27 Mabini St., Cebu City 6000", "ADDRESS") == "[ADDRESS in Cebu City]"


def test_synthetic_values_do_not_link_documents_and_are_not_derived_from_the_original():
    t = Transformer("synthesize", seed=5)
    outs = set()
    for _ in range(30):
        outs.add(t.apply("mail maria@example.com", [__import__("veilbench.schema", fromlist=["Span"]).Span(5, 22, "EMAIL")]))
    assert len(outs) > 25  # a new value every document
    assert all("maria" not in o for o in outs)


def test_synthetic_replacement_is_consistent_inside_one_document():
    from veilbench.schema import Span
    text = "a@example.com and again a@example.com"
    spans = [Span(0, 13, "EMAIL"), Span(24, 37, "EMAIL")]
    out = Transformer("synthesize", seed=1).apply(text, spans)
    first, second = re.findall(r"\S+@\S+", out)
    assert first == second


def test_unknown_technique_is_refused():
    with pytest.raises(ValueError):
        Transformer("scramble")


def test_a_pseudonym_is_redrawn_when_it_would_be_a_real_name(monkeypatch):
    """With a tiny syllable set the made-up words collide with real names constantly; the guard must redraw every time."""
    from veilbench.corpus import names as N
    from veilbench.transform import ops
    from veilbench.transform.xdoc import fold
    monkeypatch.setattr(ops, "SYLLABLES", ["ma", "ria", "jo", "hn", "an", "na", "ta", "mi", "ke", "li"])
    monkeypatch.setattr(ops, "SUFFIXES", ["son", "ton"])
    real = {fold(t) for pool in N.POOLS.values() for k in ("first", "last") for t in pool[k]}
    t = Transformer("pseudonymize")
    redrawn = 0
    for i in range(400):
        word = t._pseudo_token(f"digest-{i:04d}".encode(), "first")
        assert fold(word) not in real, word
        redrawn += fold(t._pseudo_token(f"digest-{i:04d}".encode(), "first")) == fold(word)
    assert redrawn == 400  # and the redraw is deterministic: the same digest gives the same word
