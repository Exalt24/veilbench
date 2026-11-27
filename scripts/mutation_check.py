"""Mutation check: break one guard at a time and prove the test suite notices.

A test that stays green when its guard is deleted proves nothing. For each mutation below this script
  1. copies the repository to a temporary folder (the working tree is never edited),
  2. runs the relevant tests UNMUTATED and requires them to pass, so a red result later means something,
  3. rewrites one source file in the copy, runs only the tests that should catch it, and requires a genuine TEST failure
     (a syntax error or a crashed runner does not count as caught),
  4. reports N of N caught and exits 1 if any survived.

    python scripts/mutation_check.py                 # all mutations
    python scripts/mutation_check.py privacy         # only the mutations in one area
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

V = "veilbench/"
# (area, label, file, old text, new text, test selectors). old and new may be lists when one mutation needs several edits.
MUTATIONS = [
    ("detect", "a card number no longer needs a valid Luhn digit", V + "detect/rules.py", "13 <= len(digits) <= 19 and ids.luhn_valid(digits) and not _overlaps_any", "13 <= len(digits) <= 19 and True and not _overlaps_any", ["tests/test_rules_and_merge.py"]),
    ("detect", "an IBAN no longer needs valid check digits", V + "detect/rules.py", "while v and not ids.iban_valid(v):", "while v and False:", ["tests/test_rules_and_merge.py"]),
    ("detect", "a Spanish DNI no longer needs its check letter", V + "detect/rules.py", "if ids.dni_valid(m.group(0)) and not", "if True and not", ["tests/test_rules_and_merge.py"]),
    ("detect", "a version string is read as an IP address", V + "detect/rules.py", "if VERSION_CONTEXT.search(text[max(0, m.start() - 20):m.start()]):\n                continue", "if False:\n                continue", ["tests/test_rules_and_merge.py"]),
    ("detect", "a lower case commit hash is read as a secret", V + "detect/rules.py", "if not (any(ch.isdigit() for ch in v) and any(ch.isupper() for ch in v)):", "if False:", ["tests/test_rules_and_merge.py"]),
    ("detect", "any date is a birth date", V + "detect/rules.py", "if DOB_CONTEXT.search(before) and not", "if True and not", ["tests/test_rules_and_merge.py"]),
    ("detect", "a bare digit run (an epoch timestamp) is a phone number", V + "detect/rules.py", "if re.fullmatch(r\"\\d+\", v) and not v.startswith((\"+\", \"0\")):\n                continue", "if False:\n                continue", ["tests/test_rules_and_merge.py"]),
    ("detect", "the e-mail words form stops at the first reserved top level name", V + "detect/rules.py", "(?: dot (?:{TLDS}))*\\b", "\\b", ["tests/test_rules_and_merge.py"]),
    ("detect", "a stop word after a cue is taken as a name", V + "detect/rules.py", "if first in STOP_NAMES:", "if False:", ["tests/test_rules_and_merge.py"]),
    ("detect", "the merge keeps a sloppy copy of a finding", V + "detect/base.py", "elif new_len >= keep_fraction * (s.end - s.start):", "elif new_len >= 0:", ["tests/test_rules_and_merge.py"]),
    ("detect", "the merge lets a heuristic outrank a validated rule", V + "detect/base.py", "PRIORITY = [\"rules\", \"gliner\", \"stat\", \"spacy\", \"llm\"]", "PRIORITY = [\"llm\", \"spacy\", \"stat\", \"gliner\", \"rules\"]", ["tests/test_rules_and_merge.py"]),
    ("ids", "the Luhn check doubles the wrong digits", V + "ids.py", "        if i % 2 == 1:\n            n *= 2", "        if i % 2 == 0:\n            n *= 2", ["tests/test_ids_and_fpe.py", "tests/test_rules_and_merge.py"]),
    ("ids", "an IBAN passes on a wrong remainder", V + "ids.py", "% 97 == 1", "% 97 != 0", ["tests/test_ids_and_fpe.py"]),
    ("fpe", "FF1 runs 8 rounds instead of 10", V + "transform/fpe.py", ["    for i in range(10):\n        m = u if i % 2 == 0 else v\n        c = (_num(a) + y_for(i, bb))"], ["    for i in range(8):\n        m = u if i % 2 == 0 else v\n        c = (_num(a) + y_for(i, bb))"], ["tests/test_ids_and_fpe.py"]),
    ("fpe", "a masked card is no longer Luhn valid", V + "transform/fpe.py", "    fixed = ids.luhn_complete(digits, len(pos))\n    out[pos[-1]] = fixed[-1]", "    pass", ["tests/test_ids_and_fpe.py", "tests/test_transforms.py"]),
    ("fpe", "masked separators move", V + "transform/fpe.py", "    for i, dgt in zip(body, enc):\n        out[i] = str(dgt)\n    return \"\".join(out)\n\n\ndef mask_card", "    return \"\".join(str(d) for d in enc)\n\n\ndef mask_card", ["tests/test_ids_and_fpe.py"]),
    ("transform", "masking keeps every digit instead of the last four", V + "transform/ops.py", "keep = set(idx[-4:]) if label != \"NATIONAL_ID\" else set()", "keep = set(idx)", ["tests/test_transforms.py"]),
    ("transform", "masking keeps the first four digits of a secret too", V + "transform/ops.py", "    if label in SECRET_TYPES - {\"CREDIT_CARD\"}:\n        return \"\".join(ch if ch.isspace() else \"*\" for ch in surface)", "    if label in SECRET_TYPES - {\"CREDIT_CARD\"}:\n        return surface[:4] + \"*\" * (len(surface) - 4)", ["tests/test_transforms.py"]),
    ("transform", "the pseudonym key is ignored", V + "transform/ops.py", "return hmac.new(self.key, ", "return hmac.new(b\"fixed\", ", ["tests/test_transforms.py"]),
    ("transform", "the e-mail pseudonym is built from the surface form, not the clean address", V + "transform/ops.py", "d = self._h(\"email\", canon_email(surface)).hex()", "d = self._h(\"email\", surface).hex()", ["tests/test_transforms.py"]),
    ("transform", "a label with no pseudonym rule (a secret) passes through unchanged", V + "transform/ops.py", "        return f\"[{label}]\"  # secrets (keys, tokens, passwords) have no stand-in: they are removed, never pseudonymized", "        return surface", ["tests/test_transforms.py"]),
    ("transform", "a pseudonym name can be a real name from the pools", V + "transform/ops.py", "            if fold(word) not in REAL_TOKENS:\n                return word", "            if True:\n                return word", ["tests/test_transforms.py"]),
    ("transform", "synthetic values are linkable across documents", V + "transform/ops.py", "            self._synth_memo = {}  # a fresh value per document: nothing links two documents", "            pass", ["tests/test_transforms.py"]),
    ("transform", "a synthetic stand-in may contain a real name from the document", V + "transform/ops.py", "            if not ({fold(t) for t in re.findall(r\"\\w+\", out)} & self._avoid):\n                break", "            break", ["tests/test_transforms.py"]),
    ("transform", "anonymize keeps the full birth date", V + "transform/ops.py", "return f\"{m.group(1)}{m.group(2)}0s\" if m else \"[DATE_OF_BIRTH]\"", "return surface", ["tests/test_transforms.py"]),
    ("xdoc", "every similarity clears the threshold", V + "transform/xdoc.py", "if (fam_ok is None or fam_ok) and (giv_ok is None or giv_ok) and (fam_ok or giv_ok):", "if True:", ["tests/test_xdoc_and_structured.py"]),
    ("xdoc", "an ambiguous mention is linked to the first candidate", V + "transform/xdoc.py", "        elif len(hits) > 1:\n            c = self._found(m)\n            c.matchable = False", "        elif len(hits) > 1:\n            c = hits[0]", ["tests/test_xdoc_and_structured.py"]),
    ("xdoc", "a stand-alone cluster can be joined by later mentions", V + "transform/xdoc.py", "            if not c.matchable:\n                continue\n", "", ["tests/test_xdoc_and_structured.py"]),
    ("privacy", "the k-anonymity split no longer keeps k rows on each side", V + "privacy/kanon.py", ["            if len(left) >= k and len(right) >= k:\n                split(left)"], ["            if len(left) >= 1 and len(right) >= 1:\n                split(left)"], ["tests/test_privacy.py"]),
    ("privacy", "measured k reports the largest group", V + "privacy/kanon.py", "return min(Counter(keys).values()) if keys else 0", "return max(Counter(keys).values()) if keys else 0", ["tests/test_privacy.py"]),
    ("privacy", "Laplace noise is biased", V + "privacy/dp.py", "return rng.laplace(0.0, scale, size)", "return rng.laplace(0.5, scale, size)", ["tests/test_privacy.py"]),
    ("privacy", "the noise scale ignores epsilon", V + "privacy/dp.py", "noisy = counts.astype(float) + laplace(rng, 1.0 / epsilon, counts.shape)", "noisy = counts.astype(float) + laplace(rng, 1.0, counts.shape)", ["tests/test_privacy.py"]),
    ("privacy", "a non-positive epsilon is accepted", V + "privacy/dp.py", "    if epsilon <= 0:\n        raise ValueError(\"epsilon must be positive\")\n    noisy", "    noisy", ["tests/test_privacy.py"]),
    ("privacy", "the attack counts a row as re-identified when several names fit", V + "privacy/linkage.py", "        if len(cands) == 1 and cands[0] == r.pid:", "        if len(cands) >= 1 and r.pid in cands:", ["tests/test_privacy.py"]),
    ("metrics", "F2 weights precision and recall equally", V + "evaluation/metrics.py", "    b2 = beta * beta", "    b2 = 1.0", ["tests/test_metrics_and_benchmark.py"]),
    ("metrics", "a surviving surname is not a leak", V + "evaluation/metrics.py", "return any(len(r) >= 2 and r.lower() not in PARTICLES for r in runs)", "return False", ["tests/test_metrics_and_benchmark.py"]),
    ("metrics", "the verbatim check matches inside longer words", V + "evaluation/metrics.py", "r\"(?<![a-z0-9])\" + re.escape(needle.lower()) + r\"(?![a-z0-9])\"", "re.escape(needle.lower())", ["tests/test_metrics_and_benchmark.py"]),
    ("metrics", "a hard negative inside a real identifier counts as a false alarm", V + "evaluation/metrics.py", "and not any(p.overlaps(g) for g in rec.spans)", "", ["tests/test_metrics_and_benchmark.py"]),
    ("structured", "keys are never cleaned", V + "structured.py", "new_key = self._clean(k) if isinstance(k, str) else k", "new_key = k", ["tests/test_xdoc_and_structured.py"]),
    ("structured", "a string that is itself JSON is not opened", V + "structured.py", "if t[:1] in \"{[\" and t[-1:] in \"}]\":", "if False:", ["tests/test_xdoc_and_structured.py"]),
    ("structured", "field names are no longer a hint", V + "structured.py", "if hint and (self.use_hints or voted) and stripped.lower()", "if False and stripped.lower()", ["tests/test_xdoc_and_structured.py"]),
    ("structured", "a failing structured path is not caught", V + "structured.py", "            except Exception as e:  # noqa: BLE001 - fail closed: any failure falls through to the text pass", "            except KeyboardInterrupt as e:", ["tests/test_xdoc_and_structured.py"]),
    ("corpus", "the escaped JSON string shifts its spans by the wrong amount", V + "corpus/generate.py", "add_start = before.count('\"') + before.count(\"\\\\\")", "add_start = 0", ["tests/test_corpus.py"]),
    ("corpus", "the train split can use the held out cultures", V + "corpus/generate.py", "cultures = N.TRAIN_CULTURES if train else", "cultures = list(N.ALL_CULTURES) if train else", ["tests/test_corpus.py"]),
]


def copy_repo(dst: Path) -> None:
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache", ".hypothesis", ".work", "docs", "*.egg-info"))


def run_tests(repo: Path, selectors: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-W", "ignore", *selectors]
    env = {"PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1", **{k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}}
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, env=env)
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


def genuine_failure(out: str) -> str | None:
    """The name of a failing test, or None when the run failed for another reason (syntax error, import error, a crash)."""
    if re.search(r"SyntaxError|IndentationError|ImportError|ModuleNotFoundError|NameError: name .* is not defined|error during collection", out):
        return None
    m = re.search(r"^FAILED (\S+)", out, re.M)
    return m.group(1) if m else None


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    todo = [m for m in MUTATIONS if not only or m[0] == only]
    tmp = Path(tempfile.mkdtemp(prefix="veilbench-mutation-"))
    repo = tmp / "copy"
    try:
        copy_repo(repo)
        print(f"working in a copy: {repo}\n\nbaseline (unmutated tests must pass):")
        for sel in dict.fromkeys(tuple(m[5]) for m in todo):
            ok, out = run_tests(repo, list(sel))
            print(f"  {'pass' if ok else 'FAIL'}  {' '.join(sel)}", flush=True)
            if not ok:
                print(out[-1500:])
                print("\nBASELINE IS RED: fix the tests before trusting any mutation result.")
                return 2
        survived, caught = [], 0
        print("\nmutations:")
        for area, label, rel, old, new, selectors in todo:
            path = repo / rel
            original = path.read_text(encoding="utf-8")
            olds = old if isinstance(old, (list, tuple)) else [old]
            news = new if isinstance(new, (list, tuple)) else [new]
            if any(o not in original for o in olds):
                print(f"  MISSING   [{area}] {label}: the text to mutate is not in {rel}")
                survived.append(label + " (text not found)")
                continue
            mutated = original
            for o, n in zip(olds, news):
                mutated = mutated.replace(o, n, 1)
            start = time.time()
            try:
                path.write_text(mutated, encoding="utf-8", newline="\n")
                passed, out = run_tests(repo, selectors)
            finally:
                path.write_text(original, encoding="utf-8", newline="\n")
            failing = None if passed else genuine_failure(out)
            if passed:
                verdict, note = "SURVIVED", ""
            elif failing is None:
                verdict, note = "INVALID ", "  (failed, but not as a test failure: syntax error or crashed runner)"
            else:
                verdict, note = "caught  ", f"  <- {failing.split('::', 1)[-1][:80]}"
            print(f"  {verdict}  [{area}] {label}  ({time.time() - start:.0f}s){note}", flush=True)
            if verdict == "caught  ":
                caught += 1
            else:
                survived.append(label)
        ok, _ = run_tests(repo, list(todo[-1][5]))
        print(f"\n{caught} of {len(todo)} caught, {len(survived)} survived or invalid; the copy is back to green: {ok}")
        if survived:
            print("PROBLEMS:", "; ".join(survived))
            return 1
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
