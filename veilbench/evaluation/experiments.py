"""The measured experiments that are not the detector benchmark: privacy (k-anonymity, DP, linkage attack), utility, cross document names, robustness."""
from __future__ import annotations

import random
import re
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from veilbench.corpus import people as P
from veilbench.corpus.generate import generate_corpus, ticket_corpus
from veilbench.detect.base import merge_spans
from veilbench.evaluation.metrics import verbatim_leak
from veilbench.privacy import dp, kanon, linkage, synth
from veilbench.privacy.population import make_population
from veilbench.schema import Record, Span
from veilbench.structured import StructuredAnonymizer
from veilbench.transform.ops import TECHNIQUES, Transformer
from veilbench.transform.xdoc import NameResolver, fold

# ------------------------------------------------------------------ privacy: k-anonymity, differential privacy, linkage


def privacy_report(seed: int = 11) -> dict:
    pop = make_population(3000, seed)
    rng = random.Random(seed)
    released = [p for p in pop if rng.random() < 0.5]
    raw = kanon.raw_keys(released)
    out: dict = {"population": len(pop), "released_rows": len(released), "attacker_named_list": len(pop)}
    out["before"] = {"measured_k": kanon.measured_k(raw), "unique_row_share": round(kanon.unique_share(raw), 4), **linkage.linkage_exact(released, pop)}

    ks = []
    for k in (2, 5, 10, 25):
        gen = kanon.mondrian(released, k)
        keys = [g.key() for g in gen]
        ks.append({"k_requested": k, "measured_k": kanon.measured_k(keys), "unique_row_share": round(kanon.unique_share(keys), 4),
                   "information_loss": round(kanon.info_loss(gen, released), 4), **linkage.linkage_generalized(gen, pop)})
    out["k_anonymity"] = ks

    eps = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    out["dp_count_error_curve"] = dp.epsilon_error_curve(eps)
    out["membership_attack"] = linkage.membership_attack(pop, [None] + eps)
    syn = []
    for e in (1.0, 5.0):
        rows = synth.synthesize(pop, e, seed=9)
        syn.append({"epsilon": e, **linkage.synthetic_false_claims(rows, pop), **synth.marginal_error(pop, rows)})
    out["synthetic"] = syn
    return out


# ------------------------------------------------------------------ utility: a classifier and a join key, before and after


def _oracle(rec: Record) -> list[Span]:
    return rec.spans


def _key_of(text: str) -> str:
    m = re.search(r"Account email: (.+)", text)
    return m.group(1).strip() if m else ""


def _classify(train: list[tuple[str, str]], test: list[tuple[str, str]]) -> float:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([t for t, _ in train])
    clf = LogisticRegression(max_iter=400).fit(X, [y for _, y in train])
    pred = clf.predict(vec.transform([t for t, _ in test]))
    return float(np.mean([p == y for p, (_, y) in zip(pred, test)]))


def utility_report(detectors: dict[str, callable] | None = None, seed: int = 21) -> dict:
    """`detectors` maps a name to text -> spans. The oracle (ground truth spans) isolates what a technique itself does to utility."""
    tickets = ticket_corpus(250, 1200, seed)
    labels = [t.meta["category"] for t in tickets]
    split = int(len(tickets) * 0.7)
    truth_distinct = len({t.meta["customer"] for t in tickets})
    true_counts = Counter(t.meta["customer"] for t in tickets)
    truth_repeat = sum(1 for c in true_counts.values() if c >= 2)

    def evaluate(name: str, texts: list[str]) -> dict:
        acc = _classify(list(zip(texts[:split], labels[:split])), list(zip(texts[split:], labels[split:])))
        keys = [_key_of(t) for t in texts]
        kc = Counter(keys)
        distinct = len(kc)
        repeat = sum(1 for c in kc.values() if c >= 2)
        return {"name": name, "classifier_accuracy": round(acc, 4), "distinct_customers": distinct,
                "distinct_customers_error": round(abs(distinct - truth_distinct) / truth_distinct, 4),
                "repeat_customers": repeat, "repeat_customers_error": round(abs(repeat - truth_repeat) / truth_repeat, 4)}

    rows = [evaluate("raw text (no anonymization)", [t.text for t in tickets])]
    for tech in TECHNIQUES:
        tr = Transformer(tech, seed=1)
        texts = [tr.apply(t.text, _oracle(t)) for t in tickets]
        rows.append(evaluate(f"{tech} (ground truth spans)", texts))
    for name, det in (detectors or {}).items():
        tr = Transformer("pseudonymize")
        texts = [tr.apply(t.text, merge_spans(det(t.text))) for t in tickets]
        row = evaluate(f"pseudonymize ({name})", texts)
        row["verbatim_leaking_tickets"] = sum(1 for t, o in zip(tickets, texts) if verbatim_leak(t, o))
        rows.append(row)
    return {"tickets": len(tickets), "customers": 250, "train": split, "test": len(tickets) - split, "true_distinct_customers": truth_distinct,
            "true_repeat_customers": truth_repeat, "rows": rows}


# ------------------------------------------------------------------ techniques compared on one corpus


def technique_table(records: list[Record]) -> list[dict]:
    """Every technique applied with perfect detection to the benchmark corpus: what is left of the identifiers, and what each one keeps."""
    props = {
        "redact": dict(reversible_by_key_holder="no", keeps_join=False, keeps_format=False),
        "mask": dict(reversible_by_key_holder="no", keeps_join=False, keeps_format=True),
        "pseudonymize": dict(reversible_by_key_holder="numbers by decryption, names only from a stored map", keeps_join=True, keeps_format=True),
        "anonymize": dict(reversible_by_key_holder="no", keeps_join=False, keeps_format=False),
        "synthesize": dict(reversible_by_key_holder="no", keeps_join=False, keeps_format=True),
    }
    out = []
    for t in TECHNIQUES:
        tr = Transformer(t, seed=2)
        leaking = sum(1 for r in records if verbatim_leak(r, tr.apply(r.text, r.spans)))
        out.append({"technique": t, "records": len(records), "verbatim_leaking_records": leaking, **props[t]})
    return out


# ------------------------------------------------------------------ cross document names


RESOLVABLE = ["full", "last_first", "upper", "accent_stripped", "typo", "initial_last", "first_initial"]


def _variants(rng: random.Random, p: P.Person) -> dict[str, str | None]:
    g, f = " ".join(p.given), " ".join(p.family)
    typo = None
    if len(f) > 4:
        i = rng.randrange(1, len(f) - 1)
        typo = f"{g} {f[:i]}{f[i + 1:]}"  # a dropped letter in the family name
    return {"full": f"{g} {f}", "last_first": f"{f}, {g}", "upper": f"{g} {f}".upper(), "accent_stripped": f"{P.strip_accents(g)} {P.strip_accents(f)}",
            "typo": typo, "initial_last": f"{g[0]}. {f}", "first_initial": f"{g} {f[0]}.", "title_last": f"Ms. {f}"}


def xdoc_report(thresholds=(0.80, 0.85, 0.90, 0.93, 0.96, 0.98), n_people: int = 300, seed: int = 4) -> dict:
    """Mentions of each person arrive in shuffled order across documents. Three questions per threshold:

    1. Do the variants a name can reasonably take (spelling, order, case, accents, one dropped letter, an initial) land in one cluster?
    2. Do near twins (same family name, a given name one letter apart) stay two people?
    3. A bare "Ms. Last" fits many people: is it linked to the wrong one (a false merge), or left alone?
    """
    rng = random.Random(seed)
    people, seen = [], set()
    while len(people) < n_people:
        p = P.make_person(rng, ["en", "es", "ph", "de"])
        if fold(p.full) not in seen and len(p.given[0]) > 3:
            seen.add(fold(p.full))
            people.append(p)
    twins = [P.Person(**{**p.__dict__, "given": [p.given[0][:-1] + ("o" if p.given[0][-1] != "o" else "a")]}) for p in people[::10]]
    out = []
    for th in thresholds:
        r2 = random.Random(seed)
        res = NameResolver(th)
        items = []  # (person key, kind, text)
        for i, p in enumerate(people):
            items += [(("p", i), k, v) for k, v in _variants(r2, p).items() if v]
        for j, p in enumerate(twins):
            items += [(("t", j), k, v) for k, v in _variants(r2, p).items() if k in ("full", "last_first", "upper")]
        r2.shuffle(items)
        res.fit([text for _, _, text in items])
        cluster_of = [(pk, kind, res.resolve(text).cid) for pk, kind, text in items]
        resolvable = [(pk, cid) for pk, kind, cid in cluster_of if kind in RESOLVABLE]
        main = {}
        for pk in {pk for pk, _ in resolvable}:
            main[pk] = Counter(cid for q, cid in resolvable if q == pk).most_common(1)[0]
        in_main = sum(n for _, n in main.values())
        split = sum(1 for pk in main if len({cid for q, cid in resolvable if q == pk}) > 1)
        by_cluster = defaultdict(set)
        for pk, cid in resolvable:
            by_cluster[cid].add(pk)
        merged = sum(1 for s_ in by_cluster.values() if len(s_) > 1)
        twin_merged = sum(1 for j in range(len(twins)) if main[("t", j)][0] == main.get(("p", j * 10), (None,))[0])
        amb = [(pk, cid) for pk, kind, cid in cluster_of if kind == "title_last"]
        attached_right = sum(1 for pk, cid in amb if cid == main.get(pk, (None,))[0])
        other_mains = {c for q, (c, _) in main.items()}
        attached_wrong = sum(1 for pk, cid in amb if cid != main.get(pk, (None,))[0] and cid in other_mains)
        out.append({"threshold": th, "people": len(main), "resolvable_mentions": len(resolvable),
                    "mentions_in_their_person_main_cluster": round(in_main / len(resolvable), 4), "people_split_over_clusters": split,
                    "clusters_merging_different_people": merged, "near_twin_pairs": len(twins), "near_twin_pairs_merged": twin_merged,
                    "title_last_mentions": len(amb), "title_last_linked_to_right_person": attached_right, "title_last_linked_to_a_person_cluster_wrongly": attached_wrong})
    return {"people": n_people, "near_twin_people": len(twins), "rows": out, "chosen_threshold": 0.96}


# ------------------------------------------------------------------ robustness: schema aware versus flat


def robustness_report(records: list[Record], detect, tag: str = "") -> dict:
    """Verbatim leakage of the JSON and CSV records with the structure aware anonymizer and with a flat text pass."""
    rows = {}
    for label, flat in (("structure aware", False), ("flat text pass", True)):
        tr = Transformer("redact")
        by = defaultdict(lambda: [0, 0])
        t0 = time.perf_counter()
        for r in records:
            if r.kind not in ("json", "csv"):
                continue
            if flat:
                out = tr.apply(r.text, merge_spans(detect(r.text)))
            else:
                out = StructuredAnonymizer(detect, tr).run(r.text).text
            leaked = bool(verbatim_leak(r, out))
            for sl in set(r.slices) & {"json", "csv", "drift", "unexpected_field", "obfuscated"}:
                by[sl][0] += int(leaked)
                by[sl][1] += 1
        rows[label] = {sl: {"leaking": v[0], "records": v[1], "rate": round(v[0] / v[1], 4)} for sl, v in sorted(by.items())}
        rows[label]["seconds"] = round(time.perf_counter() - t0, 2)
    return rows


