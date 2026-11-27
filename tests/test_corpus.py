import json
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from veilbench.corpus.generate import corpus_sha, corpus_stats, generate_corpus, ticket_corpus
from veilbench.schema import ENTITY_TYPES


def test_the_default_corpus_meets_the_size_the_benchmark_claims():
    stats = corpus_stats(generate_corpus(800, 7))
    assert stats["records"] == 800 and stats["spans"] >= 2000
    assert set(stats["spans_by_type"]) == set(ENTITY_TYPES)
    for slice_ in ("drift", "unexpected_field", "obfuscated", "non_english", "hard_negative", "clean"):
        assert stats["records_by_slice"][slice_] >= 50, slice_


def test_same_seed_same_corpus_different_seed_different_corpus():
    assert corpus_sha(generate_corpus(60, 5)) == corpus_sha(generate_corpus(60, 5))
    assert corpus_sha(generate_corpus(60, 5)) != corpus_sha(generate_corpus(60, 6))


@settings(max_examples=12, deadline=None)
@given(st.integers(min_value=0, max_value=10_000), st.sampled_from(["eval", "train", "ood"]))
def test_every_span_is_inside_the_text_and_none_overlap(seed, split):
    for rec in generate_corpus(25, seed, split):
        spans = sorted(rec.spans)
        for s in spans:
            assert 0 <= s.start < s.end <= len(rec.text) and s.label in ENTITY_TYPES
        for a, b in zip(spans, spans[1:]):
            assert a.end <= b.start, (rec.id, rec.text[a.start:b.end])
        for n in rec.negatives:
            assert not any(n.overlaps(s) for s in spans)


def test_secret_spans_are_exactly_the_secret_even_inside_an_escaped_json_string(corpus):
    """A key placed inside a JSON string inside a JSON field must still be found at its real offsets after the quote escaping."""
    seen = 0
    for rec in corpus:
        for s in rec.spans:
            if s.label in ("API_KEY", "JWT"):
                value = rec.text[s.start:s.end]
                assert re.fullmatch(r"[A-Za-z0-9_\-.]+", value), (rec.kind, value)
                seen += 1
    assert seen >= 40


def test_json_records_parse_and_nested_payloads_hold_the_secret(corpus):
    nested = 0
    for rec in corpus:
        if rec.kind != "json":
            continue
        data = json.loads(rec.text)
        payload = data.get("payload")
        if isinstance(payload, str):
            assert json.loads(payload)["key"]
            nested += 1
    assert nested >= 5


def test_obfuscated_spans_keep_the_clean_value_for_the_leak_check(corpus):
    found = 0
    for rec in corpus:
        for s in rec.spans:
            if s.label == "EMAIL" and " dot " in rec.text[s.start:s.end]:
                assert "@" in s.canon and " " not in s.canon
                found += 1
    assert found >= 3


def test_train_split_never_uses_the_cultures_held_out_for_evaluation():
    names = " ".join(r.text for r in generate_corpus(150, 1, "train"))
    for held_out in ("Nguyen", "Okafor", "Sharma", "Haddad"):
        assert held_out not in names


def test_ticket_corpus_has_one_join_key_per_customer():
    tickets = ticket_corpus(40, 200, 3)
    keys = {}
    for t in tickets:
        k = re.search(r"Account email: (.+)", t.text).group(1)
        keys.setdefault(t.meta["customer"], set()).add(k)
    assert all(len(v) == 1 for v in keys.values())
    assert len({next(iter(v)) for v in keys.values()}) == len(keys)
