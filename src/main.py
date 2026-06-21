"""Main entry point for the IEEE Fraud Detection pipeline."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import constants
from utils.data_handlers import (
    column_types,
    correlation_and_variance,
    features_and_target,
    get_numeric_and_categorical_columns,
    holdout_test_split,
    load_csv_data,
    merge_transaction_and_identity,
    null_profile,
    time_series_split,
)
from utils.pipeline_tools import (
    bdl_reference,
    compare_models_on_test,
    fit_full_model,
    mlp_reference,
    run_pipeline,
    save_pipeline_params,
    xgboost_reference,
)

if TYPE_CHECKING:
    import polars as pl


def merge(
    train_transactions_df: pl.DataFrame,
    train_identity_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge transaction and identity DataFrames.

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
        train_transactions_df = load_csv_data(
            constants.TRAIN_TRANSACTIONS
        )  # [:10000]  # comment out for faster testing
        # train_identity_df = load_csv_data(constants.TRAIN_IDENTITY)#[:10000]
    except FileNotFoundError as e:
        sys.stderr.write(f"Error loading data from csv file: {e}\n")
        sys.exit(1)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")

    #####################
    # Try the merged data
    #####################
    # report_name = f"{timestamp}_merged"
    # report_dir = constants.REPORTS_DIR / report_name
    # report_dir.mkdir(parents=True, exist_ok=True)

    # merge_df = merge(train_transactions_df, train_identity_df)

    # column_types(merge_df, report_dir)
    # null_profile(merge_df, report_dir)
    # correlation_and_variance(merge_df, report_dir)

    # # NaNs as is
    # report_name = "raw_nan"
    # raw_report_dir = report_dir / report_name
    # _, merge_cat_cols = get_numeric_and_categorical_columns(
    #     merge_df,
    #     exclude_columns=[constants.TARGET, constants.TIMESTAMP],
    # )
    # merge_pipeline, merge_param_distributions = xgboost_reference(
    #     categorical_features=merge_cat_cols,
    # )
    # save_pipeline_params(merge_pipeline, raw_report_dir)
    # x, y, tscv = time_series_split(
    #     merge_df,
    #     constants.TARGET,
    #     constants.TIMESTAMP,
    # )
    # run_pipeline(
    #     merge_pipeline,
    #     merge_param_distributions,
    #     tscv,
    #     x,
    #     y,
    #     raw_report_dir,
    # )

    # # NaNs imputed
    # report_name = "imputed_nan"
    # imputed_report_dir = report_dir / report_name
    # merge_num_cols, merge_cat_cols = get_numeric_and_categorical_columns(
    #     merge_df,
    #     exclude_columns=[constants.TARGET, constants.TIMESTAMP],
    # )
    # merge_pipeline, merge_param_distributions = mlp_reference(
    #     categorical_features=merge_cat_cols, numeric_features=merge_num_cols
    # )
    # save_pipeline_params(merge_pipeline, imputed_report_dir)
    # x, y, tscv = time_series_split(
    #     merge_df,
    #     constants.TARGET,
    #     constants.TIMESTAMP,
    # )
    # run_pipeline(
    #     merge_pipeline,
    #     merge_param_distributions,
    #     tscv,
    #     x,
    #     y,
    #     imputed_report_dir,
    # )
    #####################
    # Try the merged data
    #####################

    #############################################
    # Try just transactions without the left join
    #############################################
    report_name = f"{timestamp}_transactions_only"
    report_dir = constants.REPORTS_DIR / report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    # Hold out the most recent transactions as a labeled test set, reserved for
    # later evaluation. Everything below (EDA, CV search, full refit) only sees
    # train_df; the test set is never touched during training.
    train_df, test_df = holdout_test_split(
        train_transactions_df,
        constants.TIMESTAMP,
        test_fraction=0.2,
    )
    test_df.write_parquet(report_dir / "holdout_test.parquet")

    column_types(train_df, report_dir)
    null_profile(train_df, report_dir)
    correlation_and_variance(train_df, report_dir)

    # NaNs as is: xgboost
    report_name = "xgboost_reference"
    raw_report_dir = report_dir / report_name
    _, trans_cat_cols = get_numeric_and_categorical_columns(
        train_df,
        exclude_columns=[constants.TARGET, constants.TIMESTAMP],
    )
    transactions_pipeline, transactions_param_distributions = xgboost_reference(
        categorical_features=trans_cat_cols,
    )
    save_pipeline_params(transactions_pipeline, raw_report_dir)
    x, y, tscv = time_series_split(
        train_df,
        constants.TARGET,
        constants.TIMESTAMP,
    )
    run_pipeline(
        transactions_pipeline,
        transactions_param_distributions,
        tscv,
        x,
        y,
        raw_report_dir,
    )
    # Separate full-data refit: take the best CV params and train one model on
    # all of train_df, saving it as the deployable artifact.
    xgb_model = fit_full_model(transactions_pipeline, x, y, raw_report_dir)

    # NaNs imputed: MLP
    # report_name = "mlp_reference"
    # imputed_report_dir = report_dir / report_name
    # trans_num_cols, trans_cat_cols = get_numeric_and_categorical_columns(
    #     train_transactions_df,
    #     exclude_columns=[constants.TARGET, constants.TIMESTAMP],
    # )
    # transactions_pipeline, transactions_param_distributions = mlp_reference(
    #     categorical_features=trans_cat_cols, numeric_features=trans_num_cols
    # )
    # save_pipeline_params(transactions_pipeline, imputed_report_dir)
    # x, y, tscv = time_series_split(
    #     train_transactions_df,
    #     constants.TARGET,
    #     constants.TIMESTAMP,
    # )
    # run_pipeline(
    #     transactions_pipeline,
    #     transactions_param_distributions,
    #     tscv,
    #     x,
    #     y,
    #     imputed_report_dir,
    # )

    # NaNs imputed: BDL
    report_name = "bdl_reference"
    print(f"report name: {report_name}")
    imputed_report_dir = report_dir / report_name
    trans_num_cols, trans_cat_cols = get_numeric_and_categorical_columns(
        train_df,
        exclude_columns=[constants.TARGET, constants.TIMESTAMP],
    )
    transactions_pipeline, transactions_param_distributions = bdl_reference(
        categorical_features=trans_cat_cols, numeric_features=trans_num_cols
    )
    save_pipeline_params(transactions_pipeline, imputed_report_dir)
    x, y, tscv = time_series_split(
        train_df,
        constants.TARGET,
        constants.TIMESTAMP,
    )
    run_pipeline(
        transactions_pipeline,
        transactions_param_distributions,
        tscv,
        x,
        y,
        imputed_report_dir,
    )
    # Separate full-data refit: take the best CV params and train one model on
    # all of train_df, saving it as the deployable artifact.
    bdl_model = fit_full_model(transactions_pipeline, x, y, imputed_report_dir)

    # Test step: score the deployable XGBoost and BDL models on the held-out
    # test set never seen during EDA/CV/refit. Surfaces PR-AUC and precision for
    # XGBoost, BDL without a reject mask, and BDL with a reject mask, writing the
    # side-by-side comparison to report_dir/test_comparison.json. The BDL reject
    # mask rejects test data whose Bayes Error exceeds the reference computed by
    # inferring the full training set (x); here x is exactly that training set.
    x_test, y_test = features_and_target(
        test_df,
        constants.TARGET,
        constants.TIMESTAMP,
    )

    compare_models_on_test(
        xgb_model,
        bdl_model,
        x,
        x_test,
        y_test,
        report_dir,
    )
    #############################################
    # Try just transactions without the left join
    #############################################


if __name__ == "__main__":
    sys.stdout.write("Program start...\n")
    begin = datetime.now(tz=UTC)
    main()
    end = datetime.now(tz=UTC)
    sys.stdout.write("Program stop.\n")
    sys.stdout.write(f"Elapsed runtime: {end - begin}\n")
