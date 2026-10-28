"""
fewshot_picker_clean.py — leakage-free 8-shot set with pragmatic-refutation
Refuted examples (Week 5/6 "convention-type NEE" ablation).

Why this exists (2026-07-02): fewshot_picker.py selected its 8 examples from
the main system's WRONG predictions on sample150 — but those claims ARE the
s150 evaluation set. All 8 Week 4 examples appear verbatim in s150, so the
LLM saw 8 eval claims with their gold labels inside the prompt. Recomputing
without the 8 leaked claims: fewshot8 s150 Macro 0.4092 -> 0.3416, i.e. BELOW
plain zero-shot (0.3498 on the same 142). The Week 4 dev500 numbers are
unaffected (official dev is disjoint from train.json).

This picker therefore draws ONLY from internal_train.json and additionally
excludes any claim text that also appears in internal_dev.json or
internal_dev_sample150.json (AVeriTeC contains duplicate claim texts across
splits: 4 collide with s150, 15 with internal_dev).

Composition (8 total, balanced as in Week 4):
  - 2 Supported            random, QA 3-5, seed-fixed
  - 2 Refuted              HAND-PINNED pragmatic-refutation examples that
                           demonstrate the annotation convention the gold-
                           oracle analysis showed the LLM misses (see
                           analyze_gold_gap.py): satire/fiction source =>
                           refuted, and "No answer could be found" + dis-
                           credited source => refuted (not NEE).
                           Pinned by claim_idx in internal_train.json:
                             466 — satire site story ("publishes fiction")
                             396 — fabricated quote, no record + retracted
  - 2 NEE                  random, QA 3-5, seed-fixed
  - 2 Conflicting          QA 3-5, >=1 with explicit conflict marker in the
                           gold justification (Week 4 criterion (d))

Output: data/fewshot_clean8_pragref.json (same schema as fewshot_template.json)

Usage:
    python src/fewshot_picker_clean.py
"""
import argparse
import json
import os
import random
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# ── Self-contained helpers (inlined so this file has no dependency on the
#    original leakage-affected fewshot_picker.py, which is intentionally not
#    shipped — see README) ───────────────────────────────────────────────
LABEL_NUM_TO_STR = {
    0: "supported",
    1: "refuted",
    2: "not enough evidence",
    3: "conflicting evidence/cherrypicking",
}
LABEL_STR_TO_NUM = {
    "Supported": 0,
    "Refuted": 1,
    "Not Enough Evidence": 2,
    "Conflicting Evidence/Cherrypicking": 3,
}
CONFLICT_MARKER_RE = re.compile(
    r"\b(however|but|on the other hand|contradicts?|contradicted|"
    r"while|whereas|nevertheless|nonetheless|despite|conflicting|"
    r"disputed?|disagree[sd]?)\b",
    re.IGNORECASE,
)


def format_evidence_text(record: dict) -> str:
    """Format AVeriTeC train.json `questions` field into the evidence_text
    string consumed by reprompt_eval.format_fewshot_block."""
    lines = []
    for i, q in enumerate(record.get("questions", []), 1):
        qtext = (q.get("question") or "").strip()
        for a in q.get("answers", []):
            atext = (a.get("answer") or "").strip()
            if atext:
                lines.append(f"[{i}] Q: {qtext}\n    A: {atext}")
                break
    return "\n".join(lines)


PINNED_REFUTED_IDX = [466, 396]  # indices into internal_train.json


def make_example(rec: dict, idx: int, lbl: int, pinned: bool) -> dict:
    return {
        "claim": rec.get("claim", "").strip(),
        "speaker": (rec.get("speaker") or "unknown").strip(),
        "source": (rec.get("reporting_source") or "unknown").strip(),
        "evidence_text": format_evidence_text(rec),
        "label": LABEL_NUM_TO_STR[lbl],
        "_picker_meta": {
            "picker": "clean-v1 (internal_train only, eval-split overlap excluded)",
            "claim_idx_in_internal_train": idx,
            "n_questions": len(rec.get("questions", [])),
            "hand_pinned_pragmatic_refutation": pinned,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/internal_train.json")
    parser.add_argument("--exclude", nargs="+",
                        default=["data/internal_dev.json",
                                 "data/internal_dev_sample150.json"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/fewshot_clean8_pragref.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool = json.load(open(os.path.join(PROJECT_DIR, args.pool), encoding="utf-8"))
    banned = set()
    for p in args.exclude:
        banned |= {c["claim"].strip()
                   for c in json.load(open(os.path.join(PROJECT_DIR, p), encoding="utf-8"))}
    print(f"Pool: {len(pool)} claims | banned texts from eval splits: {len(banned)}")

    fewshot = []

    # Refuted: hand-pinned pragmatic-refutation demonstrations
    for idx in PINNED_REFUTED_IDX:
        rec = pool[idx]
        assert rec.get("label") == "Refuted", f"idx {idx} is not Refuted"
        assert rec["claim"].strip() not in banned, f"idx {idx} collides with an eval split"
        fewshot.append(make_example(rec, idx, 1, pinned=True))
        print(f"  pinned Refuted [{idx}]: {rec['claim'][:80]}")

    # Supported / NEE / Conflicting: seed-fixed random, QA 3-5, no eval overlap
    for lbl in (0, 2, 3):
        label_str = next(k for k, v in LABEL_STR_TO_NUM.items() if v == lbl)
        cands = [(i, rec) for i, rec in enumerate(pool)
                 if rec.get("label") == label_str
                 and 3 <= len(rec.get("questions", [])) <= 5
                 and rec["claim"].strip() not in banned
                 and format_evidence_text(rec).strip()]
        if lbl == 3:  # criterion (d): at least 1 with explicit conflict marker
            marked = [(i, r) for i, r in cands
                      if CONFLICT_MARKER_RE.search(r.get("justification") or "")]
            pick1 = rng.choice(marked)
            rest = [c for c in cands if c[0] != pick1[0]]
            pick2 = rng.choice(rest)
            chosen = [pick1, pick2]
        else:
            chosen = rng.sample(cands, 2)
        for i, rec in chosen:
            fewshot.append(make_example(rec, i, lbl, pinned=False))
            print(f"  {LABEL_NUM_TO_STR[lbl]:>20} [{i}]: {rec['claim'][:80]}")

    # order examples: Sup, Sup, Ref, Ref, NEE, NEE, Conf, Conf (Week 4 order)
    order = {"supported": 0, "refuted": 1, "not enough evidence": 2,
             "conflicting evidence/cherrypicking": 3}
    fewshot.sort(key=lambda e: order[e["label"]])

    out = os.path.join(PROJECT_DIR, args.output)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fewshot, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Written {len(fewshot)} examples to {out}")


if __name__ == "__main__":
    main()
