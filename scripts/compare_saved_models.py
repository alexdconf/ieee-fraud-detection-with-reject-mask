"""Re-run ``compare_models_on_test`` from saved ``.joblib`` models.

Loads the XGBoost and BDL pipelines and the held-out test set produced by a
previous ``main.py`` run, rebuilds the reject-mask reference with
``trouble_reference`` (or the full training set), and writes a fresh
``test_comparison.json`` -- without retraining anything.

The reference's training features are not persisted by ``main.py``, so they are
reconstructed deterministically from the source CSV via the same
``holdout_test_split`` the run used.

Examples:
    # Latest run, default reference (top 5% by Bayes Error):
    uv run python scripts/compare_saved_models.py

    # A specific run, BALD-ranked reference, top 1%:
    uv run python scripts/compare_saved_models.py reports/20260621-132004_transactions_only \\
        --metric bald --quantile 0.99

    # Full-training reference, swept over Bayes-Error multipliers
    # (1, 1.25, 1.67, 2, 2.15) -- one labelled reject-mask result each:
    uv run python scripts/compare_saved_models.py --reference all

    # No reject mask -- just XGBoost and BDL test metrics (fast, no CSV reload):
    uv run python scripts/compare_saved_models.py --reference none

    # Random-abstention baseline at 92.6% coverage (match a reject-mask run's
    # coverage to compare like-for-like; no CSV reload):
    uv run python scripts/compare_saved_models.py --reference random --quantile 0.926

    # Risk--coverage curves for every abstention rule at matched coverage, with
    # AURC and 200-resample bootstrap error bars (one MC pass, no CSV reload):
    uv run python scripts/compare_saved_models.py --reference risk_coverage \\
        --bootstrap 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import polars as pl

# main.py and the utils package import each other as top-level modules (e.g.
# ``import constants``), so src/ must be on the path before importing them.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import constants  # noqa: E402
from main import trouble_reference  # noqa: E402
from utils.data_handlers import (  # noqa: E402
    features_and_target,
    holdout_test_split,
    load_csv_data,
)
from utils.pipeline_tools import (  # noqa: E402
    bayes_error_reference,
    compare_models_on_test,
    evaluate_bdl_random_mask,
    evaluate_bdl_reject_sweep,
    evaluate_on_test,
    risk_coverage_sweep,
)

_COMPARISON_FILENAME = "test_comparison.json"
_RISK_COVERAGE_FILENAME = "risk_coverage.json"

# Reject-mask thresholds swept for the full-training-set reference: each is a
# multiplier of the reduced reference Bayes Error. 1.0 is the plain reference.
_ALL_REFERENCE_MULTIPLIERS = (1.0, 2.0, 3.0, 5.0, 10.0)


def _latest_report_dir() -> Path:
    """Return the most recent ``*_transactions_only`` report directory."""
    candidates = sorted(constants.REPORTS_DIR.glob("*_transactions_only"))
    if not candidates:
        msg = (
            f"No '*_transactions_only' run found under {constants.REPORTS_DIR}. "
            "Pass a report directory explicitly."
        )
        raise SystemExit(msg)
    return candidates[-1]


def _load_model(path: Path) -> object:
    """Load a ``best_model.joblib`` pipeline, erroring clearly if absent."""
    if not path.exists():
        msg = f"Saved model not found: {path}"
        raise SystemExit(msg)
    return joblib.load(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Run directory holding the saved models and holdout_test.parquet "
        "(defaults to the latest *_transactions_only run).",
    )
    parser.add_argument(
        "--reference",
        choices=("trouble", "all", "none", "random", "risk_coverage"),
        default="trouble",
        help="Reject-mask reference: 'trouble' (trouble_reference subset), "
        "'all' (the full training set), 'none' (no reject mask -- evaluate "
        "XGBoost and BDL on the test set and report metrics only, skipping the "
        "training-set reference reconstruction), 'random' (drop test rows "
        "uniformly at random to --quantile coverage, ignoring uncertainty -- the "
        "baseline the reject mask must beat at equal coverage), or "
        "'risk_coverage' (risk--coverage curves for random/bayes_error/"
        "predictive_entropy/bald/epistemic_var/xgb_margin at matched coverage, "
        "with AURC; one MC pass, no training-set reconstruction). Default: "
        "trouble.",
    )
    parser.add_argument(
        "--metric",
        default="bayes_error",
        help="Uncertainty metric trouble_reference ranks by (e.g. bayes_error, "
        "bald). Ignored when --reference all. Default: bayes_error.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Upper-tail cut for trouble_reference (0.95 keeps the most "
        "uncertain 5%%). For --reference random it is reinterpreted as the "
        "retained coverage fraction (0.95 keeps a random 95%%); set it to a "
        "reject-mask run's coverage to compare like-for-like. Ignored when "
        "--reference all. Default: 0.95.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the --reference random and --reference risk_coverage "
        "RNGs (random rule and bootstrap). Default: 0.",
    )
    parser.add_argument(
        "--coverages",
        type=str,
        default=None,
        help="--reference risk_coverage only: comma-separated retained fractions "
        "to evaluate every rule at (e.g. '1.0,0.95,0.9,0.8'). Default: the "
        "built-in grid down to 0.80.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="--reference risk_coverage only: test-set bootstrap resamples for "
        "AURC/risk error bars (0 = none). Default: 0.",
    )
    parser.add_argument(
        "--reduction",
        choices=("mean", "median", "max"),
        default="mean",
        help="How to reduce reference Bayes Errors to the threshold scalar. "
        "Default: mean.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for the comparison JSON (default: <report_dir>/recompare).",
    )
    return parser.parse_args()


def main() -> None:
    """Load saved models and re-run the test-set comparison."""
    args = parse_args()
    report_dir = args.report_dir or _latest_report_dir()
    out_dir = args.out or report_dir / "recompare"

    xgb_model = _load_model(report_dir / "xgboost_reference" / "best_model.joblib")
    bdl_model = _load_model(report_dir / "bdl_reference" / "best_model.joblib")

    # Held-out test set: load exactly the rows main.py reserved.
    test_path = report_dir / "holdout_test.parquet"
    if not test_path.exists():
        msg = f"Holdout test set not found: {test_path}"
        raise SystemExit(msg)
    test_df = pl.read_parquet(test_path)
    x_test, y_test = features_and_target(test_df, constants.TARGET, constants.TIMESTAMP)

    sys.stdout.write(f"Report dir:  {report_dir}\n")
    sys.stdout.write(f"Test rows:   {len(x_test)}\n")

    if args.reference == "none":
        # No reject mask: just score both models on the test set. Skips the
        # training-set reconstruction and Bayes-Error reference entirely.
        sys.stdout.write("Reference:   none (no reject mask)\n\n")
        results = {
            "xgboost": evaluate_on_test(xgb_model, x_test, y_test),
            "bdl_no_reject_mask": evaluate_on_test(bdl_model, x_test, y_test),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / _COMPARISON_FILENAME).open("w") as f:
            json.dump(results, f, indent=4)
    elif args.reference == "random":
        # Random-abstention baseline: drop test rows uniformly at random to the
        # target coverage, ignoring uncertainty. Like 'none', it needs no
        # training-set reconstruction -- the mask is drawn on the test set alone.
        sys.stdout.write(
            f"Reference:   random (coverage={args.quantile}, seed={args.seed})\n\n"
        )
        no_mask, random_mask = evaluate_bdl_random_mask(
            bdl_model, x_test, y_test, coverage=args.quantile, seed=args.seed
        )
        results = {
            "xgboost": evaluate_on_test(xgb_model, x_test, y_test),
            "bdl_no_reject_mask": no_mask,
            "bdl_random_mask": random_mask,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / _COMPARISON_FILENAME).open("w") as f:
            json.dump(results, f, indent=4)
    elif args.reference == "risk_coverage":
        # Matched-coverage risk--coverage curves for every abstention rule, from
        # one MC pass. Like 'none'/'random', no training-set reconstruction.
        coverages = (
            [float(c) for c in args.coverages.split(",")]
            if args.coverages
            else None
        )
        sys.stdout.write(
            f"Reference:   risk_coverage (bootstrap={args.bootstrap}, "
            f"seed={args.seed})\n\n"
        )
        results = risk_coverage_sweep(
            bdl_model,
            xgb_model,
            x_test,
            y_test,
            coverages=coverages,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / _RISK_COVERAGE_FILENAME).open("w") as f:
            json.dump(results, f, indent=4)
        # AURC league table (lower is better); the headline of the sweep.
        sys.stdout.write("AURC by abstention rule (balanced error, lower=better):\n")
        ranked = sorted(results["methods"].items(), key=lambda kv: kv[1]["aurc"])
        for name, method in ranked:
            boot = (
                f" +/- {method['aurc_boot_std']:.5f}"
                if "aurc_boot_std" in method
                else ""
            )
            sys.stdout.write(f"  {name:<20} {method['aurc']:.5f}{boot}\n")
        sys.stdout.write(f"\nWrote {out_dir / _RISK_COVERAGE_FILENAME}\n")
        return
    else:
        # Reject-mask path: reconstruct the training split deterministically and
        # build the reference the mask thresholds against.
        train_df, _ = holdout_test_split(
            load_csv_data(constants.TRAIN_TRANSACTIONS),
            constants.TIMESTAMP,
            test_fraction=0.2,
        )
        x_train, _ = features_and_target(
            train_df, constants.TARGET, constants.TIMESTAMP
        )
        if args.reference == "trouble":
            x_reference = trouble_reference(
                bdl_model, x_train, metric=args.metric, quantile=args.quantile
            )
            sys.stdout.write(
                f"Reference:   trouble_reference(metric={args.metric}, "
                f"quantile={args.quantile}) -> {len(x_reference)}/{len(x_train)} "
                f"rows\n"
            )
            sys.stdout.write(f"Reduction:   {args.reduction}\n\n")
            results = compare_models_on_test(
                xgb_model,
                bdl_model,
                x_reference,
                x_test,
                y_test,
                out_dir,
                reduction=args.reduction,
            )
        else:
            # all: sweep the reject threshold over multipliers of the reduced
            # full-training-set reference Bayes Error, one masked result each.
            reference = bayes_error_reference(
                bdl_model, x_train, reduction=args.reduction
            )
            sys.stdout.write(
                f"Reference:   all training -> {len(x_train)} rows; base Bayes "
                f"Error ({args.reduction}) = {reference:.6f}\n"
            )
            sys.stdout.write(f"Multipliers: {_ALL_REFERENCE_MULTIPLIERS}\n\n")
            no_mask, reject_masks = evaluate_bdl_reject_sweep(
                bdl_model, x_test, y_test, reference, _ALL_REFERENCE_MULTIPLIERS
            )
            results = {
                "xgboost": evaluate_on_test(xgb_model, x_test, y_test),
                "bdl_no_reject_mask": no_mask,
            }
            for entry in reject_masks:
                results[f"bdl_reject_mask_x{entry['multiplier']}"] = entry
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / _COMPARISON_FILENAME).open("w") as f:
                json.dump(results, f, indent=4)

    sys.stdout.write(json.dumps(results, indent=2) + "\n")
    sys.stdout.write(f"\nWrote {out_dir / _COMPARISON_FILENAME}\n")


if __name__ == "__main__":
    main()
