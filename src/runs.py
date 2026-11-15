"""
runs.py — the on-disk contract between the pipeline and the evaluation.

WHY THIS EXISTS
---------------
Until now every experiment in this repo scored itself. `bm25_baseline.py`
retrieved, prompted, parsed and computed Macro-F1 in one process, and every
new variant copied that block. Two things go wrong when scoring lives inside
the pipeline:

  1. A change to the metric silently rewrites history. There is no way to
     re-score an old run under a corrected metric without re-running the LLM.
  2. You cannot compare a run against a run you did not write, because the
     numbers are the only thing that survived.

So the pipeline now writes ARTIFACTS and nothing else; `eval_verdict.py`
reads those artifacts and is the only code allowed to compute a metric. A run directory is self-contained and re-scoreable forever:

    runs/<run_id>/
      config.yaml        what was run (every knob, plus code provenance)
      retrieved.jsonl    claim_id, rank, passage_id, score
      passages.jsonl     passage_id -> url, title, text   (deduplicated)
      predictions.jsonl  claim_id, gold_label, raw_output, used_passage_ids
      system.jsonl       claim_id, latency_ms, tokens, cost, status
      metrics.json       written by the eval scripts, never by the pipeline

THE ONE DESIGN DECISION THAT MATTERS: passage_id
------------------------------------------------
`passage_id` is a content hash of (url, text) — NOT a row number, and NOT
derived from the run that retrieved it. The same passage retrieved by any
ranker in any run therefore carries the same id, which is what makes runs
comparable passage by passage: two runs that share a ranking have
byte-identical `retrieved.jsonl` files, and a run that showed the model
different TEXT for the same passage shows up as a different id.

That second property is not hypothetical. The two published BM25 prompt runs
share a ranking, but one showed the model full chunks and the other 300-char
cached previews; hashing the text is what keeps those from looking like the
same evidence.

The cost of this choice is that a passage whose text differs by one
character is a different passage. That is the correct trade: chunking is
deterministic here (CHUNK_SIZE=200, CHUNK_STRIDE=100), so the same KS
document chunked by the same code always yields byte-identical passages.

LABELS
------
Gold labels are stored as the dataset's own strings ("Not Enough Evidence",
not 2). The integer encoding is a property of the evaluation, not of the
run, and three different scripts in this repo had three different orders.
`eval_verdict.py` owns the mapping.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 1

RUN_FILES = {
    "config": "config.yaml",
    "retrieved": "retrieved.jsonl",
    "passages": "passages.jsonl",
    "predictions": "predictions.jsonl",
    "system": "system.jsonl",
    "metrics": "metrics.json",
}

GOLD_LABELS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)


# ── passage identity ────────────────────────────────────────────────────
def passage_id(url: str, text: str) -> str:
    """
    Stable content hash for a retrieved passage.

    Deliberately insensitive to: which run retrieved it, at what rank, with
    what score, and from which retriever. Deliberately sensitive to: the
    passage text and its source URL, because those are what the model
    was actually shown.

    Normalisation is limited to stripping surrounding whitespace. Nothing
    else is folded (no lowercasing, no unicode normalisation): a passage
    that differs in case is different text.
    """
    h = hashlib.sha1()
    h.update((url or "").strip().encode("utf-8"))
    h.update(b"\n\x00\n")          # separator that cannot occur in a URL
    h.update((text or "").strip().encode("utf-8"))
    return h.hexdigest()[:16]


# ── jsonl ───────────────────────────────────────────────────────────────
def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    """Writes rows as one compact JSON object per line. Returns the count."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str) -> Iterator[dict]:
    """Streams rows. Blank lines are skipped; a bad line names itself."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e


# ── provenance ──────────────────────────────────────────────────────────
def code_provenance() -> dict:
    """
    Git state at the moment of the run.

    A run directory that cannot be traced to a commit is not reproducible,
    and `dirty: true` is the honest answer when the working tree had
    uncommitted edits — which is most of the time during development.
    """
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ("git",) + args,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        except Exception:
            return None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


# ── the run object ──────────────────────────────────────────────────────
@dataclass
class Run:
    """An in-memory run, loaded from or about to be written to a directory."""
    run_dir: str
    config: dict = field(default_factory=dict)
    retrieved: list[dict] = field(default_factory=list)
    passages: dict[str, dict] = field(default_factory=dict)
    predictions: list[dict] = field(default_factory=list)
    system: list[dict] = field(default_factory=list)

    # -- construction helpers used by the pipeline ----------------------
    def add_retrieved(self, claim_id: str, ranked: list[tuple[float, str, str]],
                      titles: list[str] | None = None) -> list[str]:
        """
        Records one claim's ranked passages, deduplicating passage text into
        `passages` and returning the passage_ids in rank order.

        `ranked` is the (score, url, text) triple shape the retrievers in
        this repo already return, so a pipeline can hand over its existing
        structure unchanged.
        """
        ids = []
        for rank, (score, url, text) in enumerate(ranked, start=1):
            pid = passage_id(url, text)
            if pid not in self.passages:
                self.passages[pid] = {
                    "passage_id": pid,
                    "url": url,
                    "title": (titles[rank - 1] if titles
                              and rank - 1 < len(titles) else None),
                    "text": text,
                }
            self.retrieved.append({
                "claim_id": claim_id,
                "rank": rank,
                "passage_id": pid,
                "score": float(score),
            })
            ids.append(pid)
        return ids

    def add_prediction(self, claim_id: str, claim: str, gold_label: str,
                       raw_output: str, used_passage_ids: list[str],
                       **extra: Any) -> None:
        row = {
            "claim_id": claim_id,
            "claim": claim,
            "gold_label": gold_label,
            "raw_output": raw_output,
            "used_passage_ids": list(used_passage_ids),
        }
        row.update(extra)
        self.predictions.append(row)

    def add_system(self, claim_id: str, **fields: Any) -> None:
        self.system.append({"claim_id": claim_id, **fields})

    # -- io -------------------------------------------------------------
    def write(self) -> dict[str, int]:
        import yaml

        os.makedirs(self.run_dir, exist_ok=True)
        cfg = dict(self.config)
        cfg.setdefault("schema_version", SCHEMA_VERSION)
        cfg.setdefault("code", code_provenance())
        with open(os.path.join(self.run_dir, RUN_FILES["config"]), "w",
                  encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        counts = {
            "retrieved": write_jsonl(
                os.path.join(self.run_dir, RUN_FILES["retrieved"]),
                sorted(self.retrieved,
                       key=lambda r: (r["claim_id"], r["rank"]))),
            "passages": write_jsonl(
                os.path.join(self.run_dir, RUN_FILES["passages"]),
                (self.passages[k] for k in sorted(self.passages))),
            "predictions": write_jsonl(
                os.path.join(self.run_dir, RUN_FILES["predictions"]),
                sorted(self.predictions, key=lambda r: r["claim_id"])),
            "system": write_jsonl(
                os.path.join(self.run_dir, RUN_FILES["system"]),
                sorted(self.system, key=lambda r: r["claim_id"])),
        }
        return counts

    # -- reading / querying ---------------------------------------------
    def ranking(self, claim_id: str) -> list[str]:
        """passage_ids for one claim in rank order."""
        rows = [r for r in self.retrieved if r["claim_id"] == claim_id]
        return [r["passage_id"] for r in sorted(rows, key=lambda r: r["rank"])]

    def rankings(self) -> dict[str, list[str]]:
        """{claim_id: [passage_id in rank order]} for every claim."""
        out: dict[str, list[tuple[int, str]]] = {}
        for r in self.retrieved:
            out.setdefault(r["claim_id"], []).append((r["rank"], r["passage_id"]))
        return {cid: [pid for _, pid in sorted(v)] for cid, v in out.items()}


def load_run(run_dir: str) -> Run:
    """Reads a run directory back. Missing optional files are tolerated."""
    import yaml

    def maybe(name: str) -> list[dict]:
        path = os.path.join(run_dir, RUN_FILES[name])
        return list(read_jsonl(path)) if os.path.exists(path) else []

    cfg_path = os.path.join(run_dir, RUN_FILES["config"])
    config = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    return Run(
        run_dir=run_dir,
        config=config,
        retrieved=maybe("retrieved"),
        passages={p["passage_id"]: p for p in maybe("passages")},
        predictions=maybe("predictions"),
        system=maybe("system"),
    )


def write_metrics(run_dir: str, section: str, payload: dict) -> str:
    """
    Merges a metrics section into metrics.json.

    Each evaluator owns one top-level section, so adding a new one later
    must not erase `eval_verdict.py`'s results.
    """
    path = os.path.join(run_dir, RUN_FILES["metrics"])
    existing = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing[section] = payload
    existing["schema_version"] = SCHEMA_VERSION
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)
    return path
