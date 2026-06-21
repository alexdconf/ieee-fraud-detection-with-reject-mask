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

    # Compare against the old "all of training" reference instead:
    uv run python scripts/compare_saved_models.py --reference all

    # No reject mask -- just XGBoost and BDL test metrics (fast, no CSV reload):
    uv run python scripts/compare_saved_models.py --reference none
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
    compare_models_on_test,
    evaluate_on_test,
)

_COMPARISON_FILENAME = "test_comparison.json"


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
        choices=("trouble", "all", "none"),
        default="trouble",
        help="Reject-mask reference: 'trouble' (trouble_reference subset), "
        "'all' (the full training set), or 'none' (no reject mask -- evaluate "
        "XGBoost and BDL on the test set and report metrics only, skipping the "
        "training-set reference reconstruction). Default: trouble.",
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
        "uncertain 5%%). Ignored when --reference all. Default: 0.95.",
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
            ref_desc = (
                f"trouble_reference(metric={args.metric}, quantile={args.quantile}) "
                f"-> {len(x_reference)}/{len(x_train)} rows"
            )
        else:
            x_reference = x_train
            ref_desc = f"all training -> {len(x_reference)} rows"

        sys.stdout.write(f"Reference:   {ref_desc}\n")
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

    sys.stdout.write(json.dumps(results, indent=2) + "\n")
    sys.stdout.write(f"\nWrote {out_dir / _COMPARISON_FILENAME}\n")


if __name__ == "__main__":
    main()
