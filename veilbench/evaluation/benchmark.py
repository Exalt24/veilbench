"""The detector benchmark: every configuration over one seeded corpus, scored on span quality, leakage, latency and cost.

The LLM pass is rate limited (a free key allows 1,000 requests a day and 8,000 tokens a minute), so it runs on the first
`llm_records` records only; every other configuration is also scored on that same subset so the rows are comparable.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

from veilbench.corpus.generate import corpus_sha, corpus_stats, generate_corpus
from veilbench.detect import llm as llm_mod
from veilbench.detect.llm import LLMReviewer
from veilbench.detect.pipeline import CONFIGS, ComponentStore, build_component, config_prediction, run_component
from veilbench.evaluation.metrics import Evaluation, percentile, verbatim_leak
from veilbench.schema import ENTITY_TYPES, Record
from veilbench.transform.ops import Transformer

LOCAL = ["rules", "stat", "spacy", "gliner"]


def _available(comp: str) -> tuple[bool, str]:
    if comp in ("rules", "stat"):
        return True, ""
    try:
        if comp == "spacy":
            import spacy
            spacy.load("en_core_web_sm")
            spacy.load("xx_ent_wiki_sm")
        elif comp == "gliner":
            import gliner  # noqa: F401
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:120]


def _score(records: list[Record], names: list[str], store: ComponentStore, llm: LLMReviewer | None, preds: list[str], tag: str) -> list[dict]:
    rows = []
    for name in names:
        comps = CONFIGS[name]
        ev = Evaluation()
        lat, tok_in, tok_out, by_slice, verb_total = [], 0, 0, {}, 0
        redact = Transformer("redact")
        for rec in records:
            spans, ms, usage = config_prediction(store, rec, comps, llm)
            ev.add_record(rec, spans)
            lat.append(ms)
            tok_in += usage["in"]
            tok_out += usage["out"]
            leaked = bool(verbatim_leak(rec, redact.apply(rec.text, spans)))
            verb_total += int(leaked)
            for sl in rec.slices:
                a = by_slice.setdefault(sl, [0, 0])
                a[0] += int(leaked)
                a[1] += 1
            preds.append(json.dumps({"rec": rec.id, "config": name, "set": tag, "spans": [[s.start, s.end, s.label] for s in spans]}))
        rep = ev.report()
        mean_ms = sum(lat) / len(lat)
        rep.update({
            "config": name, "components": comps, "records_scored": len(records),
            "latency_ms": {"p50": round(percentile(lat, 0.5), 2), "p95": round(percentile(lat, 0.95), 2), "mean": round(mean_ms, 2)},
            "cost_per_1000_records": {"compute_seconds_wall": round(mean_ms, 2), "llm_usd": round(llm_mod.cost_usd(tok_in, tok_out) / len(records) * 1000, 4),
                                      "llm_tokens_in": round(tok_in / len(records) * 1000), "llm_tokens_out": round(tok_out / len(records) * 1000)},
            "verbatim_leakage_rate": round(verb_total / len(records), 4), "verbatim_leaking_records": verb_total,
            "verbatim_leakage_by_slice": {k: {"leaking": v[0], "records": v[1], "rate": round(v[0] / v[1], 4)} for k, v in sorted(by_slice.items())},
        })
        rows.append(rep)
    return rows


def run_benchmark(n: int = 800, seed: int = 7, out: Path = Path("docs/test-runs"), work: Path = Path(".work"), configs: list[str] | None = None,
                  max_seconds: float | None = None, use_llm: bool = True, split: str = "eval", llm_records: int = 300) -> dict:
    suffix = "" if split == "eval" else f"_{split}"
    records = generate_corpus(n, seed, split)
    wanted = configs or list(CONFIGS)
    store = ComponentStore(work / f"components{suffix}.jsonl")
    t_start = time.perf_counter()
    skipped: dict[str, str] = {}
    load_seconds: dict[str, float] = {}
    pending = 0
    for comp in LOCAL:
        if not any(comp in CONFIGS[c] for c in wanted):
            continue
        ok, why = _available(comp)
        if not ok:
            skipped[comp] = why
            continue
        if all(store.get(r.id, comp) for r in records):
            continue
        t0 = time.perf_counter()
        det = build_component(comp)
        load_seconds[comp] = round(time.perf_counter() - t0, 2)
        budget = None if max_seconds is None else max(1.0, max_seconds - (time.perf_counter() - t_start))
        run_component(comp, det, records, store, budget)
        pending += sum(1 for r in records if not store.get(r.id, comp))
    if pending:
        return {"incomplete": True, "components_pending_records": pending, "note": "run the same command again: finished records are kept in the work folder"}

    runnable = [c for c in wanted if not any(x in skipped for x in CONFIGS[c] if x != "llm")]
    no_llm = [c for c in runnable if "llm" not in CONFIGS[c]]
    with_llm = [c for c in runnable if "llm" in CONFIGS[c]]
    preds: list[str] = []
    rows = _score(records, no_llm, store, None, preds, "all")
    subset_rows: list[dict] = []
    llm = LLMReviewer(out / "llm_cache.jsonl") if use_llm and with_llm else None
    if llm is not None and (llm.api_key or llm.cache):
        subset = records[:llm_records]
        subset_rows = _score(subset, no_llm + with_llm, store, llm, preds, "llm_subset")
    elif with_llm:
        skipped["llm"] = "no GROQ_API_KEY and no cached responses"

    result = {
        "corpus": {"seed": seed, "split": split, **corpus_stats(records), "sha256": corpus_sha(records)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "llm_model": llm_mod.MODEL if use_llm else None, "llm_price_usd_per_million": {"input": llm_mod.PRICE_IN, "output": llm_mod.PRICE_OUT},
                        "model_load_seconds": load_seconds, "stat_training": "1500 records, seed 1, split train"},
        "skipped": skipped, "configs": rows, "llm_subset": {"records": min(llm_records, n) if subset_rows else 0, "configs": subset_rows},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"benchmark{suffix}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / f"predictions{suffix}.jsonl").write_text("\n".join(preds) + "\n", encoding="utf-8")
    (out / f"benchmark{suffix}.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def _summary_table(c: list[dict]) -> list[str]:
    L = ["| configuration | precision | recall | F1 | F2 | span caught rate | records leaking (coverage) | records leaking (verbatim) | hard negatives flagged | p50 ms | p95 ms | compute s per 1,000 records | LLM USD per 1,000 records |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in c:
        o = x["overall"]
        L.append(f"| {x['config']} | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} | {o['f2']:.3f} | {x['caught_rate']:.3f} | {x['leakage_rate']:.3f} | {x['verbatim_leakage_rate']:.3f} | "
                 f"{x['hard_negatives']['rate']:.3f} | {x['latency_ms']['p50']} | {x['latency_ms']['p95']} | {x['cost_per_1000_records']['compute_seconds_wall']} | {x['cost_per_1000_records']['llm_usd']} |")
    return L


def render_markdown(r: dict) -> str:
    c = r["corpus"]
    L = ["# Detector benchmark" + (" on formats the statistical detector never trained on" if c["split"] == "ood" else ""), "",
         f"Corpus: {c['records']} generated records, {c['spans']} labelled spans, {c['negatives']} hard negatives, seed {c['seed']}, split `{c['split']}`, sha256 `{c['sha256'][:16]}`.", ""]
    L += _summary_table(r["configs"])
    for title, key in (("Recall by entity type", "recall"), ("Precision by entity type", "precision"), ("F2 by entity type", "f2")):
        L += ["", f"## {title}", "", "| type | " + " | ".join(x["config"] for x in r["configs"]) + " |", "|---|" + "---|" * len(r["configs"])]
        for t in ENTITY_TYPES:
            L.append(f"| {t} | " + " | ".join(f"{x['per_type'].get(t, {}).get(key, 0):.3f}" for x in r["configs"]) + " |")
    slices = sorted({s for x in r["configs"] for s in x["per_slice"]})
    L += ["", "## Recall by slice (typed match, IoU 0.5)", "", "| slice | " + " | ".join(x["config"] for x in r["configs"]) + " |", "|---|" + "---|" * len(r["configs"])]
    for s in slices:
        L.append(f"| {s} | " + " | ".join(f"{x['per_slice'].get(s, {}).get('recall', 0):.3f}" for x in r["configs"]) + " |")
    L += ["", "## Records leaking by slice (verbatim: an identifier can still be read in the redacted output)", "", "| slice | records | " + " | ".join(x["config"] for x in r["configs"]) + " |", "|---|---|" + "---|" * len(r["configs"])]
    for s in slices:
        n = next((x["verbatim_leakage_by_slice"][s]["records"] for x in r["configs"] if s in x["verbatim_leakage_by_slice"]), 0)
        L.append(f"| {s} | {n} | " + " | ".join(f"{x['verbatim_leakage_by_slice'].get(s, {}).get('rate', 0):.3f}" for x in r["configs"]) + " |")
    sub = r.get("llm_subset", {})
    if sub.get("configs"):
        L += ["", f"## The LLM second pass, on the first {sub['records']} records only", "",
              "The model is rate limited, so this table scores every configuration on the same subset. Latency for the LLM row is the measured API round trip.", ""]
        L += _summary_table(sub["configs"])
    if r["skipped"]:
        L += ["", "Skipped: " + "; ".join(f"{k} ({v})" for k, v in r["skipped"].items())]
    return "\n".join(L) + "\n"
