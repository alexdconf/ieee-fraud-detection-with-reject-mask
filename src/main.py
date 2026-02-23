"""Main entry point for the IEEE Fraud Detection pipeline."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import constants
from utils.data_handlers import (
    column_types,
    correlation_and_variance,
    get_numeric_and_categorical_columns,
    load_csv_data,
    merge_transaction_and_identity,
    null_profile,
    time_series_split,
)
from utils.pipeline_tools import (
    pipeline_nans_passthrough,
    run_pipeline,
    save_pipeline_params,
)

if TYPE_CHECKING:
    import polars as pl


def init(
    train_transactions_df: pl.DataFrame,
    train_identity_df: pl.DataFrame,
) -> pl.DataFrame:
    """Initialize the data by merging transaction and identity DataFrames.

    Args:
        train_transactions_df: The transaction DataFrame.
        train_identity_df: The identity DataFrame.

    Returns:
        The merged DataFrame.

    """
    return merge_transaction_and_identity(train_transactions_df, train_identity_df)


def main() -> None:
    """Run the main pipeline for IEEE Fraud Detection."""
    try:
        train_transactions_df = load_csv_data(constants.TRAIN_TRANSACTIONS)
        train_identity_df = load_csv_data(constants.TRAIN_IDENTITY)
    except FileNotFoundError as e:
        sys.stderr.write(f"Error loading data from csv file: {e}\n")
        sys.exit(1)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")

    # Try the merged data
    report_name = f"{timestamp}_merged"
    report_dir = constants.REPORTS_DIR / report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    merge_df = init(train_transactions_df, train_identity_df)

    column_types(merge_df, report_dir)
    null_profile(merge_df, report_dir)
    correlation_and_variance(merge_df, report_dir)

    _, merge_cat_cols = get_numeric_and_categorical_columns(merge_df)
    # Target should not be in categorical features
    if constants.TARGET in merge_cat_cols:
        merge_cat_cols.remove(constants.TARGET)
    # Timestamp should also be excluded from features in time_series_split
    if constants.TIMESTAMP in merge_cat_cols:
        merge_cat_cols.remove(constants.TIMESTAMP)

    merge_pipeline, merge_param_distributions = pipeline_nans_passthrough(
        categorical_features=merge_cat_cols,
    )
    save_pipeline_params(merge_pipeline, report_dir)
    x, y, tscv = time_series_split(
        merge_df,
        constants.TARGET,
        constants.TIMESTAMP,
    )
    run_pipeline(
        merge_pipeline,
        merge_param_distributions,
        tscv,
        x,
        y,
        report_dir,
    )

    # Try just transactions without the left join
    report_name = f"{timestamp}_transactions_only"
    report_dir = constants.REPORTS_DIR / report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    column_types(train_transactions_df, report_dir)
    null_profile(train_transactions_df, report_dir)
    correlation_and_variance(train_transactions_df, report_dir)

    _, trans_cat_cols = get_numeric_and_categorical_columns(train_transactions_df)
    if constants.TARGET in trans_cat_cols:
        trans_cat_cols.remove(constants.TARGET)
    if constants.TIMESTAMP in trans_cat_cols:
        trans_cat_cols.remove(constants.TIMESTAMP)

    transactions_pipeline, transactions_param_distributions = pipeline_nans_passthrough(
        categorical_features=trans_cat_cols,
    )
    save_pipeline_params(transactions_pipeline, report_dir)
    x, y, tscv = time_series_split(
        train_transactions_df,
        constants.TARGET,
        constants.TIMESTAMP,
    )
    run_pipeline(
        transactions_pipeline,
        transactions_param_distributions,
        tscv,
        x,
        y,
        report_dir,
    )


if __name__ == "__main__":
    sys.stdout.write("Program start...\n")
    begin = datetime.now(tz=UTC)
    main()
    end = datetime.now(tz=UTC)
    sys.stdout.write("Program stop.\n")
    sys.stdout.write(f"Elapsed runtime: {end - begin}\n")
