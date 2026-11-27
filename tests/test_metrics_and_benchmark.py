import json

import pytest

from veilbench.corpus.generate import generate_corpus
from veilbench.detect.pipeline import CONFIGS, ComponentStore, config_prediction, run_component
from veilbench.detect.rules import RuleDetector
from veilbench.evaluation.benchmark import render_markdown, run_benchmark
from veilbench.evaluation.metrics import Counts, Evaluation, leaks, match_record, percentile, prf, verbatim_leak
from veilbench.schema import Record, Span

DOC = "Name: Ana Cruz, phone 0917 555 0142, home 27 Mabini St., Cebu City 6000."


def span(needle: str, label: str) -> Span:
    i = DOC.index(needle)
    return Span(i, i + len(needle), label)


def test_f2_weights_recall_more_than_f1():
    p, r, f1 = prf(tp=50, fp=50, fn=0)          # perfect recall, half precision
    _, _, f2 = prf(tp=50, fp=50, fn=0, beta=2)
    assert (p, r) == (0.5, 1.0) and f2 > f1
    _, _, f1b = prf(tp=50, fp=0, fn=50)         # perfect precision, half recall
    _, _, f2b = prf(tp=50, fp=0, fn=50, beta=2)
    assert f2b < f1b
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_typed_matching_needs_the_right_label_and_enough_overlap():
    gt = [span("Ana Cruz", "PERSON")]
    assert match_record(gt, [span("Ana Cruz", "PERSON")])["PERSON"].tp == 1
    wrong = match_record(gt, [span("Ana Cruz", "ADDRESS")])
    assert wrong["PERSON"].fn == 1 and wrong["ADDRESS"].fp == 1
    short = match_record(gt, [Span(gt[0].start, gt[0].start + 2, "PERSON")])
    assert short["PERSON"].fn == 1 and short["PERSON"].fp == 1
    twice = match_record(gt, [span("Ana Cruz", "PERSON"), span("Ana Cruz", "PERSON")])
    assert twice["PERSON"].tp == 1 and twice["PERSON"].fp == 1  # one prediction per truth


def test_leak_rules_per_type():
    person = span("Ana Cruz", "PERSON")
    assert not leaks(person, [person], DOC)
    assert leaks(person, [span("Ana", "PERSON")], DOC)           # a surname survives
    assert not leaks(Span(0, 9, "PERSON"), [Span(0, 9, "PERSON")], "dela Cruz")
    phone = span("0917 555 0142", "PHONE")
    assert not leaks(phone, [Span(phone.start, phone.start + 10, "PHONE")], DOC)  # three digits left
    assert leaks(phone, [Span(phone.start, phone.start + 5, "PHONE")], DOC)       # seven digits left
    addr = span("27 Mabini St., Cebu City 6000", "ADDRESS")
    assert leaks(addr, [span("27", "ADDRESS")], DOC)
    assert not leaks(addr, [Span(addr.start, addr.end - 4, "ADDRESS")], DOC)
    email = Span(0, 17, "EMAIL")
    assert leaks(email, [Span(0, 8, "EMAIL")], "ana.cruz@example.com")


def test_verbatim_leak_is_a_whole_word_match_and_reads_the_clean_form():
    rec = Record("x", "t", "Li wrote from li.wu@example.com", [Span(0, 2, "PERSON"), Span(14, 31, "EMAIL", canon="li.wu@example.com")], [], [])
    assert verbatim_leak(rec, "[PERSON] wrote from [EMAIL]") == []
    assert [s.label for s in verbatim_leak(rec, "Lisa wrote from [EMAIL]")] == []   # "Li" inside "Lisa" is not the person
    assert [s.label for s in verbatim_leak(rec, "Li wrote from [EMAIL]")] == ["PERSON"]
    assert "EMAIL" in [s.label for s in verbatim_leak(rec, "[PERSON] wrote from li.wu@example.com")]
    phone = Record("y", "t", "call 9 1 7 5 5 5 0 1 4 2", [Span(5, 24, "PHONE", canon="9175550142")], [], [])
    assert verbatim_leak(phone, "call 9175550142")
    assert not verbatim_leak(phone, "call 917")


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5 and percentile([5], 0.95) == 5 and percentile([], 0.5) == 0


def test_evaluation_counts_hard_negative_hits_and_leaking_records():
    rec = Record("a", "t", DOC, [span("Ana Cruz", "PERSON")], [span("Cebu City", "NEG")], ["ticket", "clean"])
    ev = Evaluation()
    ev.add_record(rec, [span("Cebu City", "ADDRESS")])
    rep = ev.report()
    assert rep["hard_negatives"] == {"total": 1, "flagged": 1, "rate": 1.0}
    assert rep["leaking_records"] == 1 and rep["per_slice"]["clean"]["fn"] == 1
    assert Counts(1, 1, 1).summary()["f1"] == 0.5


def test_a_hard_negative_inside_a_real_identifier_is_not_a_false_alarm():
    rec = Record("a", "t", DOC, [span("27 Mabini St., Cebu City 6000", "ADDRESS")], [span("Cebu City", "NEG")], [])
    ev = Evaluation()
    ev.add_record(rec, [span("27 Mabini St., Cebu City 6000", "ADDRESS")])
    assert ev.report()["hard_negatives"]["flagged"] == 0


# ---- the benchmark runner end to end, rules only

def test_benchmark_runs_end_to_end_and_is_reproducible(tmp_path):
    a = run_benchmark(n=120, seed=3, out=tmp_path / "a", work=tmp_path / "wa", configs=["rules"], use_llm=False)
    b = run_benchmark(n=120, seed=3, out=tmp_path / "b", work=tmp_path / "wb", configs=["rules"], use_llm=False)
    row = a["configs"][0]
    assert row["config"] == "rules" and a["corpus"]["records"] == 120
    assert row["overall"] == b["configs"][0]["overall"] and a["corpus"]["sha256"] == b["corpus"]["sha256"]
    assert 0 < row["overall"]["recall"] < 1 and row["leakage_rate"] > 0.2     # rules alone miss names
    assert set(row["latency_ms"]) == {"p50", "p95", "mean"} and row["latency_ms"]["p95"] >= row["latency_ms"]["p50"]
    for name in ("benchmark.json", "benchmark.md", "predictions.jsonl"):
        assert (tmp_path / "a" / name).exists()
    assert json.loads((tmp_path / "a" / "benchmark.json").read_text(encoding="utf-8"))["configs"][0]["config"] == "rules"
    assert "| rules |" in render_markdown(a)


def test_a_stopped_component_run_resumes_instead_of_starting_over(tmp_path):
    recs = generate_corpus(40, 2)
    store = ComponentStore(tmp_path / "c.jsonl")
    det = RuleDetector()
    assert run_component("rules", det, recs, store, max_seconds=0.0) == 1          # stops after the first record
    assert run_component("rules", det, recs, ComponentStore(tmp_path / "c.jsonl")) == 39
    assert run_component("rules", det, recs, ComponentStore(tmp_path / "c.jsonl")) == 0


def test_configuration_latency_is_the_sum_of_its_components(tmp_path):
    recs = generate_corpus(5, 2)
    store = ComponentStore(tmp_path / "c.jsonl")
    store.put(recs[0].id, "rules", [], 2.0)
    store.put(recs[0].id, "stat", [], 3.0)
    spans, ms, usage = config_prediction(store, recs[0], CONFIGS["rules+stat"])
    assert spans == [] and 5.0 <= ms < 5.5 and usage == {"in": 0, "out": 0}
    with pytest.raises(KeyError):
        config_prediction(store, recs[0], ["rules", "gliner"])
