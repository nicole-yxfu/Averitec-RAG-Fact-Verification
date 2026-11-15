"""
ci_gate.py — a cheap, deterministic check that a candidate configuration
has not BROKEN anything before it is allowed near the deployed service.

WHAT IT IS FOR, AND WHAT IT IS NOT
----------------------------------
30 claims cannot detect a one-point Macro-F1 regression. At n = 30 the
standard error of accuracy is about 0.09, and with five NEE and five
Conflicting claims a single flipped verdict moves a class F1 by 0.2. Any
gate that claimed otherwise would be a random number generator with a
threshold on it.

What 30 fixed claims CAN do, in seconds and with no API calls, is catch
the failures that actually happened in this project:

  * a prompt edit that wrecks the output format, so the parser returns
    UNPARSEABLE and every claim is scored wrong              -> unparseable
  * a system that collapses onto one label (the plain prompt said NEE on
    116 of 500 claims; a worse prompt says it on all of them) -> collapse
  * a component that silently does nothing (the NaN cross-encoder that
    turned re-ranking into the identity function)            -> identical
  * a run that skipped claims or crashed half-way           -> coverage
  * a gross accuracy drop                                    -> accuracy

Real comparisons go through `compare_runs.py` on dev150 / dev500 with an
exact McNemar test. The gate decides whether a candidate is worth that
comparison, not whether it is better.

THE GATE SET
------------
`data/ci_gate_30.json`: 30 dev claims, built once with a fixed seed and
committed. Stratified with the rare classes OVERSAMPLED (12 Refuted, 8
Supported, 5 NEE, 5 Conflicting vs. a natural 18/7/2/2) — a proportional
sample would contain two NEE claims, and NEE collapse is the single most
important thing to catch.

USAGE
-----
    python src/ci_gate.py --build-set                 # once; commit the file
    python src/ci_gate.py                             # every candidate in ci/gate.yaml
    python src/ci_gate.py --candidate runs/foo        # one ad-hoc candidate
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_verdict import (SHORT_NAMES, UNPARSEABLE, gold_to_int,  # noqa: E402
                          normalize_llm_output, score_predictions)
from runs import GOLD_LABELS, load_run  # noqa: E402

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_SET = os.path.join(REPO_DIR, "data", "ci_gate_30.json")
GATE_CONFIG = os.path.join(REPO_DIR, "ci", "gate.yaml")

GATE_QUOTA = {"Refuted": 12, "Supported": 8,
              "Not Enough Evidence": 5,
              "Conflicting Evidence/Cherrypicking": 5}
GATE_SEED = 0

DEFAULT_LIMITS = {
    "max_accuracy_drop": 0.15,     # ~4-5 of 30 claims
    "max_unparseable": 1,
    "max_single_label_share": 0.80,
}


# ── building the set ────────────────────────────────────────────────────
def build_gate_set(source_run: str, out_path: str = GATE_SET) -> list[dict]:
    """Stratified, rare-class-oversampled, fixed-seed. Run once."""
    preds = load_run(source_run).predictions
    by_class: dict[str, list[dict]] = {k: [] for k in GATE_QUOTA}
    for row in preds:
        if row.get("gold_label") in by_class:
            by_class[row["gold_label"]].append(row)

    rng = np.random.default_rng(GATE_SEED)
    chosen = []
    for label in GOLD_LABELS:
        pool = sorted(by_class[label], key=lambda r: r["claim_id"])
        idx = rng.choice(len(pool), size=GATE_QUOTA[label], replace=False)
        chosen += [{"claim_id": pool[i]["claim_id"], "gold_label": label,
                    "claim": pool[i].get("claim", "")} for i in sorted(idx)]

    payload = {
        "description": "Fixed 30-claim CI gate set. Do not edit by hand; "
                       "rebuild with `python src/ci_gate.py --build-set`.",
        "source_run": os.path.relpath(source_run, REPO_DIR),
        "seed": GATE_SEED,
        "quota": GATE_QUOTA,
        "claims": chosen,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return chosen


def load_gate_ids(path: str = GATE_SET) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [c["claim_id"] for c in json.load(f)["claims"]]


# ── checking a candidate ────────────────────────────────────────────────
def _on_gate(run_dir: str, gate_ids: list[str]) -> tuple[list[dict], list[str]]:
    rows = {r["claim_id"]: r for r in load_run(run_dir).predictions}
    missing = [c for c in gate_ids if c not in rows]
    return [rows[c] for c in gate_ids if c in rows], missing


def check(candidate: str, baseline: str, gate_ids: list[str],
          limits: dict = DEFAULT_LIMITS) -> dict:
    """Returns {'passed': bool, 'checks': [...], 'scores': {...}}."""
    cand_rows, missing = _on_gate(candidate, gate_ids)
    base_rows, base_missing = _on_gate(baseline, gate_ids)
    if base_missing:
        raise SystemExit(f"baseline {baseline} lacks gate claims "
                         f"{base_missing[:5]} — the gate set is stale")

    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    add("coverage", not missing,
        f"{len(gate_ids) - len(missing)}/{len(gate_ids)} gate claims present"
        + (f"; missing {missing[:5]}" if missing else ""))
    if missing:
        return {"passed": False, "checks": checks, "scores": {}}

    cand = score_predictions(cand_rows, n_boot=200)
    base = score_predictions(base_rows, n_boot=200)

    add("unparseable", cand["n_unparseable"] <= limits["max_unparseable"],
        f"{cand['n_unparseable']} unparseable replies "
        f"(limit {limits['max_unparseable']})")

    preds = [normalize_llm_output(r.get("raw_output", "")) for r in cand_rows]
    counts = {SHORT_NAMES[i]: preds.count(i) for i in range(4)}
    top_label, top_n = max(counts.items(), key=lambda kv: kv[1])
    share = top_n / len(preds)
    add("label_collapse", share <= limits["max_single_label_share"],
        f"most frequent label {top_label} = {share:.0%} of predictions "
        f"(limit {limits['max_single_label_share']:.0%}); "
        f"gold max is {max(GATE_QUOTA.values()) / len(gate_ids):.0%}")

    drop = base["accuracy"] - cand["accuracy"]
    add("accuracy", drop <= limits["max_accuracy_drop"],
        f"{cand['accuracy']:.3f} vs baseline {base['accuracy']:.3f} "
        f"(drop {drop:+.3f}, limit {limits['max_accuracy_drop']:.2f})")

    # A candidate whose replies are identical to the baseline's on every
    # gate claim is either the baseline itself or a component that did
    # nothing. Not a failure — but it must be said out loud, because that
    # is exactly what the NaN cross-encoder looked like.
    same = sum(1 for a, b in zip(cand_rows, base_rows)
               if normalize_llm_output(a.get("raw_output", ""))
               == normalize_llm_output(b.get("raw_output", "")))
    identical = (same == len(gate_ids)
                 and os.path.realpath(candidate) != os.path.realpath(baseline))

    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "identical_to_baseline": identical,
        "agreement_with_baseline": same / len(gate_ids),
        "scores": {"candidate": {k: cand[k] for k in
                                 ("accuracy", "macro_f1", "pred_dist")},
                   "baseline": {k: base[k] for k in
                                ("accuracy", "macro_f1", "pred_dist")}},
    }


def _load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--build-set", action="store_true",
                    help="(Re)build data/ci_gate_30.json from the baseline.")
    ap.add_argument("--config", default=GATE_CONFIG)
    ap.add_argument("--candidate", nargs="*", default=None,
                    help="Override the candidate list in the config.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    baseline = os.path.join(REPO_DIR, cfg["baseline"])
    limits = {**DEFAULT_LIMITS, **(cfg.get("limits") or {})}

    if args.build_set:
        chosen = build_gate_set(baseline)
        print(f"Wrote {GATE_SET}: {len(chosen)} claims")
        return

    gate_ids = load_gate_ids()
    candidates = (args.candidate if args.candidate is not None
                  else [os.path.join(REPO_DIR, c)
                        for c in cfg.get("candidates") or []])
    if not candidates:
        print("No candidates listed; nothing to gate.")
        return

    print(f"CI gate: {len(gate_ids)} fixed claims, baseline "
          f"{os.path.relpath(baseline, REPO_DIR)}\n")
    all_ok = True
    for cand in candidates:
        res = check(cand, baseline, gate_ids, limits)
        all_ok &= res["passed"]
        print(f"{'PASS' if res['passed'] else 'FAIL'}  "
              f"{os.path.relpath(cand, REPO_DIR)}")
        for c in res["checks"]:
            print(f"   [{'ok' if c['passed'] else 'XX'}] {c['check']:<15} "
                  f"{c['detail']}")
        if res.get("identical_to_baseline"):
            print("   [!!] every gate verdict is identical to the baseline — "
                  "check the new component actually ran")
        elif "agreement_with_baseline" in res:
            print(f"   [..] agrees with baseline on "
                  f"{res['agreement_with_baseline']:.0%} of gate claims")
        if res["scores"]:
            s = res["scores"]
            print(f"   [..] gate Macro-F1 {s['candidate']['macro_f1']:.3f} vs "
                  f"{s['baseline']['macro_f1']:.3f} (informational only — "
                  f"n = 30 cannot resolve this)")
        print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
