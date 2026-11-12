"""
plot_risk_coverage.py — render the risk-coverage figure for the README.

Reads the JSON written by selective_prediction.py and draws accuracy as a
function of coverage, against the flat line that abstaining at random would
give. The flat line is the whole point of the figure: a confidence score is
only worth anything to the extent the curve sits above it.

Usage:
    python src/plot_risk_coverage.py \\
        --report results/selective_prediction_dev500.json \\
        --out docs/risk_coverage.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")          # headless: never try to open a window
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", default="docs/risk_coverage.png")
    ap.add_argument("--operating-point", type=float, default=0.7,
                    help="Coverage to annotate as the proposed operating "
                         "point.")
    args = ap.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)
    summaries = report["summaries"]
    baseline = summaries[0]["overall_accuracy"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)

    ax.axhline(baseline, color="#b02a1f", ls="--", lw=1.4,
               label=f"Answer everything / abstain at random ({baseline:.1%})")

    styles = [("#1f4e79", "o", "-"), ("#6a8caf", "s", "--")]
    for summ, (color, marker, ls) in zip(summaries, styles):
        xs = [p["coverage"] for p in summ["points"]]
        ys = [p["accuracy"] for p in summ["points"]]
        ax.plot(xs, ys, marker=marker, ls=ls, color=color, lw=1.8, ms=4.5,
                label=f"{summ['name'].split('(')[0].strip()} "
                      f"(AURC {summ['aurc']:.3f})")

    # Annotate the proposed operating point on the fitted curve.
    pts = summaries[0]["points"]
    op = min(pts, key=lambda p: abs(p["coverage"] - args.operating_point))
    ax.plot([op["coverage"]], [op["accuracy"]], marker="*", ms=16,
            color="#e08a1e", zorder=5, ls="none",
            label=f"Proposed operating point: {op['coverage']:.0%} coverage, "
                  f"{op['accuracy']:.1%} accuracy")
    ax.annotate(f"answers {op['n_answered']} claims\n"
                f"{op['errors_made']} errors\n"
                f"0 NEE, 0 Conflicting",
                xy=(op["coverage"], op["accuracy"]),
                xytext=(0.46, 0.652), fontsize=8.5, color="#444",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.9,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xlabel("Coverage — fraction of claims answered without a human")
    ax.set_ylabel("Accuracy on the claims answered")
    ax.set_title("Letting the system abstain is worth more than any "
                 "retrieval change tested", fontsize=11)
    ax.set_xlim(1.03, 0.05)        # high coverage on the left
    ax.set_ylim(0.55, 0.92)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Figure written to {args.out}")


if __name__ == "__main__":
    main()
