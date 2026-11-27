import json

import pytest

from veilbench.corpus.generate import generate_corpus
from veilbench.detect.rules import RuleDetector
from veilbench.evaluation.experiments import xdoc_report
from veilbench.evaluation.metrics import verbatim_leak
from veilbench.structured import StructuredAnonymizer, hint_for
from veilbench.transform.ops import Transformer
from veilbench.transform.xdoc import NameResolver, jaro_winkler, parse_mention

R = RuleDetector()


# ---- cross document names


def test_every_reasonable_variant_of_one_name_gets_one_pseudonym():
    t = Transformer("pseudonymize")
    variants = ["Maria Santos", "SANTOS, MARIA", "MARIA SANTOS", "María Santos", "M. Santos", "Maria S."]
    t.resolver.fit(variants)
    tokens = lambda v: [x.strip(".,").lower() for x in t.replacement(v, "PERSON").split()]
    full = set(tokens("Maria Santos"))
    assert len(full) == 2
    for v in variants[:4]:
        assert set(tokens(v)) == full, v  # the same two made-up words, whatever the order, case or accents
    initial_form, first_plus_initial = tokens("M. Santos"), tokens("Maria S.")
    assert set(initial_form[1:]) <= full and first_plus_initial[0] in full
    assert any(w.startswith(initial_form[0]) for w in full)
    assert len({t.resolver.resolve(v).cid for v in variants}) == 1


def test_close_but_different_people_stay_apart_at_the_chosen_threshold():
    r = NameResolver(0.96).fit(["Maria Santos", "Mario Santos", "Maria Santos Reyes", "Maria Santo"])
    assert r.resolve("Maria Santos").cid != r.resolve("Mario Santos").cid


def test_a_bare_surname_that_fits_two_people_is_linked_to_neither():
    r = NameResolver(0.96).fit(["Maria Santos", "Pedro Santos"])
    amb = r.resolve("Ms. Santos")
    assert amb.cid not in (r.resolve("Maria Santos").cid, r.resolve("Pedro Santos").cid)
    assert r.resolve("Ms. Santos").cid == amb.cid  # and the same text always gets the same stand-alone answer


def test_a_lone_first_name_joins_the_only_person_it_can_be():
    r = NameResolver(0.96).fit(["Pedro Mendez", "Maria Santos", "Pedro"])
    assert r.resolve("Pedro").cid == r.resolve("Pedro Mendez").cid


def test_order_of_arrival_does_not_decide_who_a_full_name_joins():
    mentions = ["M. Santos", "Maria Santos", "Santos, Maria", "Mario Santos", "SANTOS, MARIA"]
    a = NameResolver(0.96).fit(mentions)
    b = NameResolver(0.96).fit(list(reversed(mentions)))
    same = lambda r: (r.resolve("Maria Santos").cid == r.resolve("SANTOS, MARIA").cid, r.resolve("Maria Santos").cid != r.resolve("Mario Santos").cid)
    assert same(a) == same(b) == (True, True)


def test_parse_mention_forms():
    m = parse_mention("dela Cruz, Juan")
    assert (m.given, m.family, m.form) == (["juan"], ["dela", "cruz"], "last_first")
    assert parse_mention("Dr. Reyes").only == "reyes"
    assert parse_mention("J. Reyes").form == "initial_last"
    assert parse_mention("Juan dela Cruz").family == ["dela", "cruz"]
    assert jaro_winkler("santos", "santos") == 1.0 and jaro_winkler("maria", "mario") < 0.96


def test_the_published_threshold_sweep_has_the_tradeoff_it_claims():
    rep = xdoc_report(thresholds=(0.85, 0.96), n_people=120)
    low, high = rep["rows"]
    assert high["near_twin_pairs_merged"] <= low["near_twin_pairs_merged"]
    assert high["near_twin_pairs_merged"] == 0
    assert high["mentions_in_their_person_main_cluster"] >= 0.7


# ---- the schema agnostic anonymizer


def run(text, technique="redact", **kw):
    return StructuredAnonymizer(R.detect, Transformer(technique), **kw).run(text)


def test_json_value_and_key_are_cleaned_and_the_output_is_still_json():
    doc = json.dumps({"id": 7, "contact": {"mail": "ana.cruz@example.com"}, "ana.cruz@example.com": "note"})
    res = run(doc)
    out = json.loads(res.text)
    assert res.mode == "json" and "ana.cruz" not in res.text and out["id"] == 7


def test_a_secret_inside_a_json_string_inside_a_json_field_is_found():
    key = "sk_" + "live_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    inner = json.dumps({"endpoint": "https://api.vendor.example/v2", "key": key})
    res = run(json.dumps({"id": 1, "payload": inner}))
    assert key not in res.text and json.loads(json.loads(res.text)["payload"])["endpoint"].startswith("https://")


def test_renamed_keys_are_cleaned_by_value_and_by_name_hint():
    res = run(json.dumps({"f1": "Hello, my name is Ana Cruz", "Nme": "Tiger", "msisdn": "0917 555 0142", "pwd": "x9y8z7"}))
    assert "Ana Cruz" not in res.text and "0917" not in res.text
    assert "x9y8z7" not in res.text and res.by_hint >= 1  # "pwd" is a hint even though no detector reads it as a password


def test_an_open_value_is_not_replaced_by_a_hint():
    out = json.loads(run(json.dumps({"email": "", "name": "***", "phone": "n/a"})).text)
    assert out == {"email": "", "name": "***", "phone": "n/a"}


def test_name_hints_are_looked_up_by_key_not_value():
    assert hint_for("customerEmail") == "EMAIL" and hint_for("X-Auth-Token") == "API_KEY" and hint_for("created") is None


def test_csv_with_a_header_without_one_and_with_odd_delimiters():
    with_header = "order;customer_name;email\nA1;Ana Cruz;ana@example.com\nA2;Ben Diaz;ben@example.com\n"
    res = run(with_header)
    assert res.mode == "csv" and "ana@example" not in res.text and res.text.splitlines()[0] == "order;customer_name;email"
    no_header = "A1|ana@example.com|paid\nA2|ben@example.com|paid\n"
    assert "ana@example" not in run(no_header).text


def test_a_column_that_is_mostly_names_is_cleaned_for_its_quiet_rows_too():
    from veilbench.schema import Span
    known = {"Ana Cruz", "Ben Diaz", "Cy Moon"}
    detect = lambda t: [Span(0, len(t), "PERSON", "rules")] if t in known else []
    csv_text = "id,who,city\n1,Ana Cruz,x\n2,Ben Diaz,x\n3,Cy Moon,x\n4,Zed Quill,x\n"
    with_vote = StructuredAnonymizer(detect, Transformer("redact"), use_hints=False).run(csv_text).text
    without = StructuredAnonymizer(detect, Transformer("redact"), use_hints=False, column_vote=False).run(csv_text).text
    assert "Zed Quill" not in with_vote and "Zed Quill" in without  # the detector missed row 4; the column's vote did not


def test_malformed_json_falls_back_to_a_plain_text_pass_and_still_cleans():
    res = run('{"mail": "ana@example.com", ')
    assert res.mode == "text" and "ana@example.com" not in res.text


def test_plain_text_and_logs_are_cleaned():
    res = run("2026-03-01 WARN login failed for ana@example.com from 10.1.2.3")
    assert res.mode == "text" and "ana@" not in res.text and "10.1.2.3" not in res.text


def test_a_failing_structured_path_still_ends_in_a_cleaned_output(monkeypatch):
    sa = StructuredAnonymizer(R.detect, Transformer("redact"))
    monkeypatch.setattr(sa, "_json", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    res = sa.run('{"mail": "ana@example.com"}')
    assert "ana@example.com" not in res.text


def test_json_and_csv_records_of_the_corpus_leak_less_through_the_structured_path(corpus):
    recs = [r for r in corpus if r.kind in ("json", "csv") and "obfuscated" not in r.slices]
    tr = Transformer("redact")
    leaks = sum(bool(verbatim_leak(r, StructuredAnonymizer(R.detect, tr).run(r.text).text)) for r in recs)
    flat = sum(bool(verbatim_leak(r, tr.apply(r.text, __import__("veilbench.detect.base", fromlist=["merge_spans"]).merge_spans(R.detect(r.text))))) for r in recs)
    assert len(recs) > 40 and leaks <= flat


def test_a_stand_alone_cluster_made_for_an_ambiguous_mention_never_absorbs_a_new_person():
    r = NameResolver(0.96).fit(["Maria Santos", "Pedro Santos", "Ms. Santos"])
    known = {r.resolve(x).cid for x in ("Maria Santos", "Pedro Santos", "Ms. Santos")}
    assert len(known) == 3
    assert r.resolve("Juan Santos").cid not in known  # a fourth Santos is a fourth person, not the leftover "Ms. Santos"


def test_a_json_string_inside_json_is_opened_so_its_field_names_count():
    inner = json.dumps({"pwd": "x9y8z7q", "note": "hello"})
    res = run(json.dumps({"id": 1, "blob": inner}))
    assert "x9y8z7q" not in res.text and res.by_hint == 1
