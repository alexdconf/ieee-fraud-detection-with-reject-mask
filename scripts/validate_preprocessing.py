"""Head-to-head validation of the new NN preprocessing vs the old one.

Compares OLD (OrdinalEncoder + StandardScaler) against NEW (cross-fitted
TargetEncoder + StandardScaler for categoricals; QuantileTransformer for
numerics) on a single time-ordered holdout, with identical model params, so the
only thing that changes is the preprocessing. Reports transformed-matrix
diagnostics (NaNs, max abs magnitude) and PR-AUC (average precision).

Usage:
    uv run python scripts/validate_preprocessing.py --model bdl --rows 0 --epochs 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import constants  # noqa: E402
from utils.data_handlers import (  # noqa: E402
    get_numeric_and_categorical_columns,
    load_csv_data,
)
from utils.pipeline_tools import MCDropoutClassifier  # noqa: E402


def build_cat(encoder: str) -> Pipeline:
    if encoder == "ordinal":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ],
        )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(target_type="binary", random_state=42)),
            ("scaler", StandardScaler()),
        ],
    )


def build_num(scaler: str) -> Pipeline:
    if scaler == "standard":
        inner = StandardScaler()
    else:
        inner = QuantileTransformer(output_distribution="normal", random_state=42)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", inner),
        ],
    )


def build_pipe(
    model: str,
    encoder: str,
    scaler: str,
    cat_cols: list[str],
    num_cols: list[str],
    epochs: int,
) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", build_cat(encoder), cat_cols),
            ("num", build_num(scaler), num_cols),
        ],
        remainder="drop",
    )
    if model == "bdl":
        clf = MCDropoutClassifier(max_iter=epochs, random_state=42)
    else:
        clf = MLPClassifier(random_state=42, max_iter=epochs)
    return Pipeline(steps=[("preprocessor", pre), ("classifier", clf)])


def run_config(
    name: str,
    model: str,
    encoder: str,
    scaler: str,
    cat_cols: list[str],
    num_cols: list[str],
    x_train,  # noqa: ANN001
    y_train: np.ndarray,
    x_test,  # noqa: ANN001
    y_test: np.ndarray,
    epochs: int,
) -> dict:
    pipe = build_pipe(model, encoder, scaler, cat_cols, num_cols, epochs)
    t0 = time.time()
    pipe.fit(x_train, y_train)
    fit_s = time.time() - t0

    xt = pipe[:-1].transform(x_test)
    xt = np.asarray(xt, dtype=np.float64)
    n_nan = int(np.isnan(xt).sum())
    max_abs = float(np.nanmax(np.abs(xt)))

    proba = pipe.predict_proba(x_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)

    print(
        f"  {name:<32} PR-AUC={pr_auc:.4f}  max|x|={max_abs:>10.2f}  "
        f"NaNs={n_nan}  fit={fit_s:.1f}s",
    )
    return {"name": name, "pr_auc": pr_auc, "max_abs": max_abs, "n_nan": n_nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bdl", "mlp"], default="bdl")
    ap.add_argument("--rows", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    df = load_csv_data(constants.TRAIN_TRANSACTIONS)
    df = df.sort(constants.TIMESTAMP)
    if args.rows > 0:
        df = df.head(args.rows)

    num_cols, cat_cols = get_numeric_and_categorical_columns(
        df, exclude_columns=[constants.TARGET, constants.TIMESTAMP]
    )

    x = df.select([c for c in df.columns if c != constants.TARGET]).to_pandas()
    y = df.select(constants.TARGET).to_pandas().to_numpy().ravel()

    n = len(x)
    cut = int(n * 0.8)
    x_train, x_test = x.iloc[:cut], x.iloc[cut:]
    y_train, y_test = y[:cut], y[cut:]

    print(
        f"model={args.model} rows={n} (train={cut}, test={n - cut}) "
        f"epochs={args.epochs}",
    )
    print(
        f"base rate: train={y_train.mean():.4f} test={y_test.mean():.4f}  "
        f"num_cols={len(num_cols)} cat_cols={len(cat_cols)}",
    )

    old = run_config(
        "OLD (ordinal + standard)", args.model, "ordinal", "standard",
        cat_cols, num_cols, x_train, y_train, x_test, y_test, args.epochs,
    )
    new = run_config(
        "NEW (target + quantile)", args.model, "target", "quantile",
        cat_cols, num_cols, x_train, y_train, x_test, y_test, args.epochs,
    )

    delta = new["pr_auc"] - old["pr_auc"]
    print(f"  delta PR-AUC (new - old) = {delta:+.4f}")


if __name__ == "__main__":
    main()
