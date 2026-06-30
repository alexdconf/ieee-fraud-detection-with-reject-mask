"""Plot risk--coverage curves from a ``risk_coverage.json`` sweep.

Reads the JSON written by ``compare_saved_models.py --reference risk_coverage``
and renders two panels: (left) balanced error vs. coverage for every abstention
rule, and (right) an AURC league table as bars, with the ``random`` baseline
drawn as the reference every rule must beat. Bootstrap bands (``risk_lo`` /
``risk_hi``) are shaded when present.

Example:
    uv run python scripts/plot_risk_coverage.py \\
        reports/20260625-155136_transactions_only/recompare/risk_coverage.json \\
        --out supplementary_material/risk_coverage.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Style per rule: (colour, linestyle, marker). The 'confidence' family (warm,
# dashed) is what a non-Bayesian model gives for free; the 'epistemic' family
# (cool, solid) is the MC-dropout signal that has to beat it. 'random' is the
# black dashed floor; 'xgb_margin' is the cross-model baseline (green).
_STYLE = {
    "random": ("#444444", (0, (4, 2)), "o"),
    "bayes_error": ("#d62728", "--", "s"),
    "predictive_entropy": ("#ff7f0e", ":", "^"),
    "bald": ("#1f77b4", "-", "o"),
    "epistemic_var": ("#17becf", "-", "D"),
    "xgb_margin": ("#2ca02c", "-.", "v"),
}
_FALLBACK = ("#7f7f7f", "-", "o")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a risk_coverage.json produced by the risk_coverage sweep.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("supplementary_material/risk_coverage.png"),
        help="Output image path. Default: supplementary_material/risk_coverage.png.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Figure DPI. Default: 200."
    )
    return parser.parse_args()


def main() -> None:
    """Render the risk--coverage figure from the sweep JSON."""
    args = parse_args()
    data = json.loads(args.json_path.read_text())
    methods = data["methods"]
    n_test = data.get("n_test")
    risk_metric = data.get("risk_metric", "balanced_error")

    # Order legend and bars by AURC (best first); random is the reference line.
    ranked = sorted(methods.items(), key=lambda kv: kv[1]["aurc"])
    random_aurc = methods.get("random", {}).get("aurc")

    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2.1, 1]}
    )

    # --- left: risk--coverage curves ---
    for name, method in ranked:
        colour, linestyle, marker = _STYLE.get(name, _FALLBACK)
        points = method["points"]
        cov = [p["coverage"] for p in points]
        risk = [p[risk_metric] for p in points]
        label = f"{name}  (AURC {method['aurc']:.4f})"
        if method["predictions"] != "bdl":
            label += f"  [{method['predictions']} preds]"
        ax_curve.plot(
            cov, risk, color=colour, linestyle=linestyle, marker=marker,
            markersize=5, linewidth=1.8, label=label, zorder=3,
        )
        if "risk_lo" in points[0]:
            ax_curve.fill_between(
                cov,
                [p["risk_lo"] for p in points],
                [p["risk_hi"] for p in points],
                color=colour, alpha=0.15, zorder=1,
            )

    ax_curve.set_xlabel("Coverage (fraction of test transactions retained)")
    ax_curve.set_ylabel(f"{risk_metric.replace('_', ' ').title()}  (lower is better)")
    title = "Risk–coverage: BDL reject-mask abstention rules at matched coverage"
    if n_test:
        title += f"\nIEEE-CIS held-out test, n = {n_test:,}"
    ax_curve.set_title(title, fontsize=11)
    ax_curve.invert_xaxis()  # full coverage (no rejection) on the left
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(fontsize=8.5, framealpha=0.9, loc="best")

    # --- right: AURC league (lower is better) ---
    names = [n for n, _ in ranked]
    aurcs = [m["aurc"] for _, m in ranked]
    colours = [_STYLE.get(n, _FALLBACK)[0] for n in names]
    y_pos = range(len(names))
    bars = ax_bar.barh(list(y_pos), aurcs, color=colours, alpha=0.85, zorder=3)
    has_boot = any("aurc_boot_std" in m for _, m in ranked)
    if has_boot:
        ax_bar.errorbar(
            aurcs, list(y_pos),
            xerr=[m.get("aurc_boot_std", 0.0) for _, m in ranked],
            fmt="none", ecolor="black", capsize=3, zorder=4,
        )
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(names, fontsize=9)
    ax_bar.invert_yaxis()  # best rule on top
    ax_bar.set_xlabel("AURC (lower is better)")
    ax_bar.set_title("AURC league table", fontsize=11)
    ax_bar.grid(True, axis="x", alpha=0.3)
    if random_aurc is not None:
        ax_bar.axvline(
            random_aurc, color="#444444", linestyle=(0, (4, 2)), linewidth=1.5,
            zorder=2, label="random baseline",
        )
        ax_bar.legend(fontsize=8.5, loc="lower right")
    for bar, value in zip(bars, aurcs):
        ax_bar.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f" {value:.4f}", va="center", ha="left", fontsize=8,
        )
    ax_bar.set_xlim(0, max(aurcs) * 1.18)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
