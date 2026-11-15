"""
export_run.py — turn an existing result cache into a `runs/<run_id>/`
directory, so past experiments live under the same contract as future ones.

WHY BACKFILL INSTEAD OF RE-RUNNING
----------------------------------
Every headline number in this project's README came out of a JSON cache
written by a script that also scored itself. Re-running them to produce run
directories would cost real API money and would not reproduce byte-for-byte
(the open-web rung depends on what DuckDuckGo returns today). Backfilling
converts the artefacts that already exist, so the run directories describe
exactly the runs that were reported — and `eval_verdict.py` re-derives the
published Macro-F1 from them as a check that the conversion lost nothing.

THE DECOUPLING THIS MAKES VISIBLE
---------------------------------
`--retrieval-cache` and `--eval-cache` are separate flags on purpose. The
BM25 standard-prompt run and the BM25 plain-prompt run share one retrieval
cache and differ only in predictions, which is why they can be exported as
two run directories with byte-identical `retrieved.jsonl` and `0.3241` vs
`0.3644` in `metrics.json`. That pair is the clearest evidence in the repo
that a retrieval component's value is not a property of the retriever alone.

CACHE SHAPES IT UNDERSTANDS
---------------------------
    {claim_id: {claim, true_label, raw_output,
                top_passages: [[score, url, text], ...],   # optional
                claim_idx, prompt_tokens, completion_tokens, ...}}

which covers `bm25_baseline.py`, `open_web_baseline.py` and
`reprompt_eval.py` output alike.

USAGE
-----
    python src/export_run.py \\
        --run runs/bm25_dev500_plain \\
        --retrieval-cache results/bm25_baseline_official_dev_with_metadata_top5_cache.json \\
        --eval-cache results/bm25_official_dev_plain_prompt_cache.json \\
        --set retriever=bm25_ks prompt=plain split=official_dev top_k=5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs import GOLD_LABELS, Run  # noqa: E402

LEGACY_INT_TO_LABEL = {i: name for i, name in enumerate(GOLD_LABELS)}

# What the LLM saw for a BM25 run. Same ranking, different text:
#   preview300            reprompt_eval.py re-prompting from the cache: each
#                         passage cut to its first 300 characters, all five
#                         fit under the 3000-char evidence cap.
#   full_capped3000       bm25_baseline.py live: full ~1,100-1,400-char chunks,
#                         and build_user_input STOPS at the first passage that
#                         would cross 3000 chars — usually after two.
EVIDENCE_VIEWS = ("preview300", "full_capped3000")


def load_fulltext(path: str) -> dict[str, dict]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["claim_id"]] = r
    return rows


def rehydrate(claim_id: str, cached: list[tuple[float, str, str]],
              full: dict | None) -> list[tuple[float, str, str]]:
    """Swap previews for full chunks, refusing anything that does not match."""
    if not cached:
        return cached
    if full is None:
        raise SystemExit(f"{claim_id}: no full-text row; finish "
                         f"rehydrate_bm25_topk.py first")
    if not full.get("matches_published_top5"):
        raise SystemExit(f"{claim_id}: recomputed ranking does not match the "
                         f"published one ({full.get('problems')})")
    out = []
    for (s_old, u_old, t_old), (s, u, t) in zip(cached, full["ranked"]):
        if u != u_old or not t.startswith(t_old):
            raise SystemExit(f"{claim_id}: full text does not extend the "
                             f"cached preview for {u_old}")
        out.append((s_old, u, t))
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def coerce_scalar(text: str):
    """
    `--set top_k=5` should land in config.yaml as 5, not '5'.

    Config values are compared across runs (and `top_k: '5'` != `top_k: 5`
    for anything that reads them back), so the obvious scalar types are
    recovered here. Anything else stays a string.
    """
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def to_label_string(value) -> str | None:
    """Caches store gold as an int (historical) or the dataset string."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return LEGACY_INT_TO_LABEL.get(value)
    s = str(value)
    return s if s in GOLD_LABELS else None


def split_title(text: str) -> tuple[str | None, str]:
    """
    Open-web passages are stored as 'Title: ...\\nSnippet: ...'.

    The title is lifted into its own field for display, but the
    text is returned UNCHANGED: the passage_id hashes the exact text, and silently stripping a prefix here would make the same passage
    hash differently depending on which exporter touched it.
    """
    if text.startswith("Title: "):
        first, _, _ = text.partition("\n")
        return first[len("Title: "):].strip() or None, text
    return None, text


def coerce_passages(raw) -> list[tuple[float, str, str]]:
    """Accepts [[score, url, text], ...] or [{score,url,text/snippet}, ...]."""
    out = []
    for item in raw or []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            out.append((float(item[0]), str(item[1]), str(item[2])))
        elif isinstance(item, dict):
            text = item.get("text") or item.get("snippet") or ""
            out.append((float(item.get("score", 0.0)),
                        str(item.get("url", "")), str(text)))
        else:
            raise ValueError(f"unrecognised passage entry: {type(item)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", required=True, help="Output runs/<run_id>/ path")
    ap.add_argument("--eval-cache", required=True,
                    help="Cache holding true_label + raw_output.")
    ap.add_argument("--retrieval-cache", default=None,
                    help="Cache holding top_passages. Defaults to "
                         "--eval-cache when it carries passages itself.")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Config fields recorded in config.yaml.")
    ap.add_argument("--note", default=None,
                    help="One line describing what this run is.")
    ap.add_argument("--fulltext", default=None,
                    help="rehydrate_bm25_topk.py output. Replaces the 300-char "
                         "previews stored in BM25 caches with the full chunks, "
                         "so passage_ids hash the passage itself.")
    ap.add_argument("--evidence-view", choices=EVIDENCE_VIEWS, default=None,
                    help="What the LLM was actually shown. Required with "
                         "--fulltext; decides used_passage_ids.")
    args = ap.parse_args()
    if args.fulltext and not args.evidence_view:
        ap.error("--fulltext needs --evidence-view: the two published BM25 "
                 "runs showed the LLM different text for the same ranking")

    eval_path = args.eval_cache
    retr_path = args.retrieval_cache or args.eval_cache

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cache = json.load(f)
    if retr_path == eval_path:
        retr_cache = eval_cache
    else:
        with open(retr_path, "r", encoding="utf-8") as f:
            retr_cache = json.load(f)

    config = {"run_id": os.path.basename(os.path.normpath(args.run))}
    if args.note:
        config["note"] = args.note
    for pair in args.set:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        k, _, v = pair.partition("=")
        config[k] = coerce_scalar(v)
    config["source"] = {
        "exported_by": "src/export_run.py",
        "eval_cache": eval_path,
        "eval_cache_sha256": sha256_file(eval_path),
        "retrieval_cache": retr_path,
        "retrieval_cache_sha256": sha256_file(retr_path),
    }

    fulltext = load_fulltext(args.fulltext) if args.fulltext else None
    if fulltext is not None:
        config["evidence_view"] = args.evidence_view
        config["source"]["fulltext"] = args.fulltext
        config["source"]["fulltext_sha256"] = sha256_file(args.fulltext)

    run = Run(run_dir=args.run, config=config)

    n_no_passages = n_no_gold = 0
    for claim_id in sorted(eval_cache, key=lambda c: (len(c), c)):
        entry = eval_cache[claim_id]
        retr = retr_cache.get(claim_id, {})

        passages = coerce_passages(retr.get("top_passages"))
        if fulltext is not None:
            passages = rehydrate(claim_id, passages, fulltext.get(claim_id))
        if not passages:
            n_no_passages += 1
        titles = [split_title(t)[0] for _, _, t in passages]
        pids = run.add_retrieved(claim_id, passages, titles=titles)

        # used_passage_ids = what the LLM was actually shown, which for the
        # standard BM25 run is NOT the top 5.
        if fulltext is not None and args.evidence_view == "full_capped3000":
            n_seen = fulltext[claim_id]["seen_full_chunks"] if passages else 0
        else:
            n_seen = len(pids)
        used = pids[:n_seen]

        gold = to_label_string(entry.get("true_label"))
        if gold is None:
            n_no_gold += 1
        run.add_prediction(
            claim_id=claim_id,
            claim=entry.get("claim") or retr.get("claim") or "",
            gold_label=gold,
            raw_output=entry.get("raw_output", ""),
            used_passage_ids=used,
            claim_idx=entry.get("claim_idx", retr.get("claim_idx")),
        )
        run.add_system(
            claim_id=claim_id,
            latency_ms=None,                      # not recorded by the caches
            prompt_tokens=entry.get("prompt_tokens"),
            completion_tokens=entry.get("completion_tokens"),
            n_docs=retr.get("n_docs"),
            n_passages=len(passages),
            status=("content_filtered"
                    if str(entry.get("raw_output", "")).strip()
                    in ("CONTENT_FILTERED", "ERROR") else "ok"),
        )

    counts = run.write()
    print(f"Wrote {args.run}")
    for name, n in counts.items():
        print(f"  {name:<12} {n:>6} rows")
    print(f"  {'passages':<12} {len(run.passages):>6} distinct "
          f"(deduplicated across claims)")
    if n_no_passages:
        print(f"  [note] {n_no_passages} claims have no retrieved passages "
              f"in {os.path.basename(retr_path)} — retrieval-layer metrics "
              f"will be undefined for them.")
    if n_no_gold:
        print(f"  [warning] {n_no_gold} claims have no usable gold label.")


if __name__ == "__main__":
    main()
