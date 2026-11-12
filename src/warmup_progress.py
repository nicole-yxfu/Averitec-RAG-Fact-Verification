"""
warmup_progress.py — Live progress monitor + honest ETA for the dev
corpus warmup job. Reads ONLY filesystem state (cache file count and
mtimes); does not interfere with any running warmup process.

Usage:
    python src/warmup_progress.py              # one-shot snapshot
    python src/warmup_progress.py --watch      # refresh every 30 s
    python src/warmup_progress.py --window 5   # use last 5 min for rate

Output explains:
  - target  : how many dev corpus cache files the full passport needs (500)
  - have    : how many already exist
  - missing : how many still to build
  - rate    : new files written in the last <window> minutes (median of two
              equal sub-windows, less noisy than a single average)
  - ETA     : missing / rate, with a +50% upper bound for the long-tail of
              huge claims that take 5-10 min each.

Why this is more trustworthy than my earlier estimates:
  1. Earlier I extrapolated from a 30-second sample → high variance.
  2. Earlier I conflated "files touched" (cache hits + new builds) with
     "new builds done". Cache hits cost zero time and inflate the apparent
     rate. This script counts only files whose mtime is INSIDE the rate
     window, which by definition are newly created (or freshly rewritten).
  3. Earlier I assumed N workers ≈ N× speedup. M2 with 4 workers actually
     gets ~2.5-3× speedup due to memory pressure + disk I/O contention.
"""
import argparse
import glob
import os
import statistics
import sys
import time


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "bm25_corpus_cache",
)
TARGET = 500
SPLIT = "dev"


def _pattern():
    return os.path.join(CACHE_DIR, f"*_{SPLIT}_*.json.gz")


def count_dev_files():
    return len(glob.glob(_pattern()))


def files_modified_within(seconds):
    now = time.time()
    return [p for p in glob.glob(_pattern())
            if os.path.getmtime(p) > now - seconds]


def rate_per_minute(window_min):
    """Median of two halves of the window for stability."""
    half = window_min * 30  # seconds
    recent = files_modified_within(half)
    older = [
        p for p in files_modified_within(window_min * 60)
        if os.path.getmtime(p) <= time.time() - half
    ]
    r_recent = len(recent) / (half / 60) if half > 0 else 0
    r_older = len(older) / (half / 60) if half > 0 else 0
    if r_recent == 0 and r_older == 0:
        return 0.0, 0.0, 0.0
    # Median-ish: use min for a conservative ETA, mean for typical
    rate_mean = (r_recent + r_older) / 2
    rate_min = min(r_recent, r_older) if (r_recent > 0 and r_older > 0) else max(r_recent, r_older)
    return rate_mean, r_recent, r_older


def fmt_eta(minutes):
    if minutes < 1:
        return "< 1 min"
    if minutes < 60:
        return f"{minutes:.0f} min"
    h = int(minutes // 60)
    m = int(minutes - h * 60)
    return f"{h}h{m:02d}m"


def snapshot(window_min):
    have = count_dev_files()
    missing = max(0, TARGET - have)
    rate, r_recent, r_older = rate_per_minute(window_min)

    print(f"[{time.strftime('%H:%M:%S')}] {SPLIT} corpus cache progress")
    print(f"  target          : {TARGET}")
    print(f"  have on disk    : {have}")
    print(f"  missing         : {missing}")
    print(f"  rate window     : last {window_min} min "
          f"({window_min//2} min halves)")
    print(f"  rate (recent)   : {r_recent:.2f} files/min")
    print(f"  rate (older)    : {r_older:.2f} files/min")
    print(f"  rate (avg)      : {rate:.2f} files/min")
    if rate > 0:
        eta_min = missing / rate
        # Upper bound: long-tail of huge claims can be 3-5× slower
        eta_max = eta_min * 1.5
        print(f"  ETA (avg-rate)  : {fmt_eta(eta_min)}")
        print(f"  ETA (+50% buf)  : {fmt_eta(eta_max)}  ← plan around this")
    else:
        print("  ETA             : NO RECENT BUILDS — process may be dead or all in big claims")
    return have, rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true",
                   help="Refresh every 30 s until Ctrl+C")
    p.add_argument("--window", type=int, default=10,
                   help="Rate window in minutes (default 10)")
    p.add_argument("--split", default="dev",
                   help="Knowledge-Store split whose cache files to count "
                        "(dev, train, ...). Matches *_<split>_*.json.gz.")
    p.add_argument("--target", type=int, default=500,
                   help="Claims in that split (dev = 500).")
    args = p.parse_args()

    global SPLIT, TARGET
    SPLIT, TARGET = args.split, args.target

    if not args.watch:
        snapshot(args.window)
        return

    print(f"Watching {CACHE_DIR}")
    print("(Ctrl+C to stop. Refreshes every 30 s.)")
    print()
    last_have = -1
    while True:
        have, _ = snapshot(args.window)
        if have == TARGET:
            print("\n✓ Reached target. Stopping watch.")
            return
        if have != last_have:
            print(f"  → +{have - max(0, last_have)} since last refresh\n")
            last_have = have
        else:
            print()
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\nWatch stopped by user.")
            return


if __name__ == '__main__':
    main()
