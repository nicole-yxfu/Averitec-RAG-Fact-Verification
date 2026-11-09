"""
benchmark.py — latency and cost distribution of the running service.

The per-request figures quoted elsewhere come from single calls. This script
produces what a deployment actually needs: p50 / p95 latency, split into the
retrieval and LLM stages, and the mean cost per verification.

It drives the service over HTTP, so it measures the real deployed path
(FastAPI validation -> open-web retrieval -> LLM -> response), not an
in-process shortcut. Only the standard library is used, so it can be run from
anywhere, including against a container.

Cold vs warm: open-web search is disk-cached per query, so the first pass over
a set of claims exercises the COLD path and a second pass the WARM one. With
--repeat both are measured and reported separately — quoting a single latency
number for a cache-backed service would be misleading.

Usage:
    uvicorn src.service:app --port 8000          # in another shell
    python src/benchmark.py --n 20 --repeat --out results/benchmark.json
"""
import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request


def load_claims(data_file: str, n: int) -> list[str]:
    """Read the first n claim strings out of an AVeriTeC JSON split."""
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)
    claims = [item["claim"] for item in data if item.get("claim")]
    return claims[:n] if n > 0 else claims


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile — well defined for the small n used here."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, math.ceil(p / 100 * len(s)) - 1)
    return s[k]


def post_verify(url: str, claim: str, top_k: int, timeout: float) -> dict:
    """One POST /verify. Returns the parsed response, or {'error': ...}."""
    payload = json.dumps({"claim": claim, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/verify", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    wall_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
        body["wall_ms"] = (time.perf_counter() - wall_start) * 1000
        return body
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read()[:200].decode("utf-8", "replace")}
    except Exception as e:  # connection refused, timeout, ...
        return {"error": type(e).__name__, "detail": str(e)[:200]}


def run_pass(url: str, claims: list[str], top_k: int, timeout: float,
             label: str) -> dict:
    """Send every claim once and summarise the responses."""
    print(f"\n--- {label} pass over {len(claims)} claims ---")
    rows, errors = [], []
    for i, claim in enumerate(claims, 1):
        r = post_verify(url, claim, top_k, timeout)
        if "error" in r:
            errors.append({"claim": claim[:80], **r})
            print(f"  {i}/{len(claims)}  ERROR {r['error']}")
            continue
        rows.append({
            "retrieval_ms": r["latency"]["retrieval_ms"],
            "llm_ms": r["latency"]["llm_ms"],
            "total_ms": r["latency"]["total_ms"],
            "wall_ms": r["wall_ms"],
            "cost_usd": r["usage"]["cost_usd"],
            "prompt_tokens": r["usage"]["prompt_tokens"],
            "verdict": r["verdict"],
        })
        print(f"  {i}/{len(claims)}  {r['verdict']:<22} "
              f"{r['latency']['total_ms']:>8.0f} ms  ${r['usage']['cost_usd']:.6f}")

    if not rows:
        return {"label": label, "n": 0, "errors": errors}

    def stats(key):
        v = [r[key] for r in rows]
        return {"p50": round(percentile(v, 50), 1),
                "p95": round(percentile(v, 95), 1),
                "mean": round(statistics.fmean(v), 1),
                "min": round(min(v), 1),
                "max": round(max(v), 1)}

    costs = [r["cost_usd"] for r in rows]
    return {
        "label": label,
        "n": len(rows),
        "n_errors": len(errors),
        "errors": errors,
        "latency_ms": {k: stats(k) for k in
                       ("retrieval_ms", "llm_ms", "total_ms", "wall_ms")},
        "cost_usd": {
            "mean": round(statistics.fmean(costs), 6),
            "total": round(sum(costs), 6),
            "per_1k_claims": round(statistics.fmean(costs) * 1000, 3),
        },
        "mean_prompt_tokens": round(statistics.fmean(
            [r["prompt_tokens"] for r in rows]), 1),
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v)
                     for v in sorted({r["verdict"] for r in rows})},
    }


def print_markdown(results: list[dict]) -> None:
    """Emit a table ready to paste into the README."""
    print("\n" + "=" * 72)
    print("\n| Pass | n | Retrieval p50 | Retrieval p95 | LLM p50 | LLM p95 | Total p50 | Total p95 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        if not r.get("n"):
            continue
        lat = r["latency_ms"]
        print(f"| {r['label']} | {r['n']} | "
              f"{lat['retrieval_ms']['p50']:.0f} ms | {lat['retrieval_ms']['p95']:.0f} ms | "
              f"{lat['llm_ms']['p50']:.0f} ms | {lat['llm_ms']['p95']:.0f} ms | "
              f"{lat['total_ms']['p50']:.0f} ms | {lat['total_ms']['p95']:.0f} ms |")
    print("\n| Pass | Mean cost / claim | Cost / 1 000 claims | Mean prompt tokens |")
    print("|---|---|---|---|")
    for r in results:
        if not r.get("n"):
            continue
        print(f"| {r['label']} | ${r['cost_usd']['mean']:.6f} | "
              f"${r['cost_usd']['per_1k_claims']:.2f} | {r['mean_prompt_tokens']:.0f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000",
                    help="Base URL of the running service.")
    ap.add_argument("--n", type=int, default=20, help="How many claims to send.")
    ap.add_argument("--data-file", default="data/dev.json",
                    help="AVeriTeC JSON split to draw claims from.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--repeat", action="store_true",
                    help="Run a second (warm-cache) pass over the same claims.")
    ap.add_argument("--out", default=None, help="Write the JSON report here.")
    args = ap.parse_args()

    # Fail early with a clear message if the service is not up.
    try:
        with urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=10) as r:
            health = json.load(r)
        print(f"Service healthy: {health}")
    except Exception as e:
        raise SystemExit(f"Cannot reach {args.url}/health ({e}). "
                         f"Start it with: uvicorn src.service:app --port 8000")

    claims = load_claims(args.data_file, args.n)
    print(f"Benchmarking {len(claims)} claims from {args.data_file}")

    results = [run_pass(args.url, claims, args.top_k, args.timeout, "cold")]
    if args.repeat:
        results.append(run_pass(args.url, claims, args.top_k, args.timeout, "warm"))

    print_markdown(results)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"url": args.url, "n_requested": args.n,
                       "top_k": args.top_k, "passes": results}, f, indent=2)
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
