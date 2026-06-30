"""Statistical analysis of a ``risk_coverage.json`` sweep -- past the graph.

Two things the figure can't give you on its own:

1. **Is the AURC gap real?** Compares each abstention rule's AURC (Area Under the
   Risk--Coverage curve, lower is better) to a baseline rule. When the sweep stored
   the raw per-resample draws (``aurc_boot_samples``, written by recent
   ``risk_coverage_sweep`` runs) this is a *paired* bootstrap test: the rules share
   each resample, so testing ``AURC_baseline(b) - AURC_rule(b)`` per draw cancels
   the shared row-sampling noise and is far more powerful than treating the two
   AURCs as independent. For older JSONs without the raw draws it falls back to a
   conservative independent-Gaussian z-test on the stored ``aurc_boot_mean/std``
   (conservative because the true paired SE is smaller).

2. **Is it a recall/precision tradeoff?** Prints each rule's recall (fraud catch
   rate), precision, and balanced error as coverage drops, so a rule that "buys"
   precision by discarding frauds (recall collapsing) is obvious.

Example:
    uv run python scripts/analyze_risk_coverage.py \\
        reports/20260625-155136_transactions_only/recompare/risk_coverage.json \\
        --baseline random --method bald
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _normal_p(z: float) -> float:
    """Two-sided p-value for a standard-normal z."""
    return math.erfc(abs(z) / math.sqrt(2))


def paired_test(samples_rule: list, samples_base: list) -> dict:
    """Paired bootstrap test of ``rule`` vs ``base`` AURC, aligned by resample.

    ``delta = AURC_base - AURC_rule`` per resample: positive => the rule has the
    lower (better) AURC. Drops resamples where either draw is null (undefined
    AURC). Returns the mean delta, paired SE, z and normal p, the 95% percentile
    CI, and a sign-based bootstrap p (resolution 1/n_draws).
    """
    rule = np.asarray(samples_rule, dtype=float)
    base = np.asarray(samples_base, dtype=float)
    mask = np.isfinite(rule) & np.isfinite(base)
    delta = base[mask] - rule[mask]
    n = int(delta.size)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1))
    z = mean / se if se > 0 else math.inf * (1 if mean > 0 else -1)
    lo, hi = (float(v) for v in np.percentile(delta, [2.5, 97.5]))
    # Two-sided sign-based bootstrap p: twice the mass on the wrong side of 0.
    p_sign = min(1.0, 2.0 * min((delta <= 0).mean(), (delta >= 0).mean()))
    return {
        "kind": "paired",
        "n": n,
        "mean": mean,
        "se": se,
        "z": z,
        "p_norm": _normal_p(z),
        "ci": (lo, hi),
        "p_sign": float(p_sign),
    }


def independent_test(rule: dict, base: dict) -> dict:
    """Conservative independent-Gaussian fallback when raw draws are absent.

    SE assumes independence, which overstates the true paired SE (the rules are
    positively correlated through shared resamples), so the p-value is an upper
    bound -- a significant result here is significant a fortiori.
    """
    mean = base["aurc_boot_mean"] - rule["aurc_boot_mean"]
    se = math.sqrt(rule["aurc_boot_std"] ** 2 + base["aurc_boot_std"] ** 2)
    z = mean / se if se > 0 else math.inf * (1 if mean > 0 else -1)
    return {
        "kind": "independent (conservative)",
        "mean": mean,
        "se": se,
        "z": z,
        "p_norm": _normal_p(z),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path, help="Path to a risk_coverage.json.")
    parser.add_argument(
        "--baseline",
        default="random",
        help="Rule every other rule is tested against. Default: random.",
    )
    parser.add_argument(
        "--method",
        default="bald",
        help="Rule whose full precision/recall trajectory is printed. Default: bald.",
    )
    parser.add_argument(
        "--rules",
        default="random,bald,bayes_error",
        help="Comma-separated rules for the cross-rule recall table. Default: "
        "random,bald,bayes_error.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the AURC tests and print the recall/precision tables."""
    args = parse_args()
    data = json.loads(args.json_path.read_text())
    methods = data["methods"]
    if args.baseline not in methods:
        raise SystemExit(f"--baseline {args.baseline!r} not in {list(methods)}")
    has_draws = "aurc_boot_samples" in methods[args.baseline]

    # --- AURC league ---
    print(f"AURC league ({data.get('risk_metric','?')}, lower is better):")
    for name, m in sorted(methods.items(), key=lambda kv: kv[1]["aurc"]):
        boot = (
            f"  boot {m['aurc_boot_mean']:.5f} +/- {m['aurc_boot_std']:.5f}"
            if "aurc_boot_mean" in m
            else ""
        )
        print(f"  {name:<20} {m['aurc']:.5f}{boot}  [{m['predictions']} preds]")

    # --- pairwise tests vs baseline ---
    mode = "paired bootstrap" if has_draws else "independent-Gaussian (CONSERVATIVE)"
    print(f"\nAURC difference vs '{args.baseline}'  --  test: {mode}")
    print("(+delta => rule beats baseline; lower AURC)")
    base = methods[args.baseline]
    others = sorted(
        (n for n in methods if n != args.baseline),
        key=lambda n: methods[n]["aurc"],
    )
    for name in others:
        rule = methods[name]
        if has_draws and "aurc_boot_samples" in rule:
            r = paired_test(rule["aurc_boot_samples"], base["aurc_boot_samples"])
            ci = f"  95% CI [{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}]"
            psign = f"  p_sign={r['p_sign']:.3g}"
        else:
            r = independent_test(rule, base)
            ci = psign = ""
        verdict = "beats" if r["mean"] > 0 else "loses to"
        sig = "*" if r["p_norm"] < 0.05 else " "
        print(
            f"  {name:<20} d={r['mean']:+.5f}  z={r['z']:+6.2f}  "
            f"p={r['p_norm']:.3g}{sig}{ci}{psign}  ({verdict} {args.baseline})"
        )
    print("  (* p<0.05, two-sided)")

    # --- precision/recall trajectory for one rule ---
    print(f"\n'{args.method}' trajectory vs coverage:")
    print(f"{'cov':>6}{'prev':>8}{'prec':>8}{'recall':>8}{'bal_err':>9}{'band(±)':>18}")
    for p in methods[args.method]["points"]:
        band = ""
        if "risk_lo" in p:
            band = f"[{p['risk_lo']:.4f},{p['risk_hi']:.4f}]"
        print(
            f"{p['coverage']:>6.2f}{p['prevalence']:>8.4f}{p['precision']:>8.3f}"
            f"{p['recall']:>8.3f}{p['balanced_error']:>9.4f}{band:>18}"
        )

    # --- cross-rule recall (the tradeoff view) ---
    rules = [r for r in args.rules.split(",") if r in methods]
    covs = [p["coverage_requested"] for p in methods[args.baseline]["points"]]

    def cell(rule: str, cov: float, key: str) -> float:
        for p in methods[rule]["points"]:
            if abs(p["coverage_requested"] - cov) < 1e-9:
                return p[key]
        return float("nan")

    for key, label in (("recall", "RECALL (fraud catch rate)"), ("precision", "PRECISION")):
        print(f"\n{label} by rule:")
        print(f"{'cov':>6}" + "".join(f"{r:>16}" for r in rules))
        for c in covs:
            print(f"{c:>6.2f}" + "".join(f"{cell(r, c, key):>16.3f}" for r in rules))


if __name__ == "__main__":
    main()
