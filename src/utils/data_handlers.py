"""Utility functions for data loading and processing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
from sklearn.model_selection import TimeSeriesSplit

if TYPE_CHECKING:
    import numpy as np
    from pandas import DataFrame as pdDataFrame


def load_csv_data(filepath: Path | str) -> pl.DataFrame:
    """Load a CSV file into a Polars DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        The loaded Polars DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    path = Path(filepath)
    if not path.is_file():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    return pl.read_csv(path)


def merge_transaction_and_identity(
    df_transaction: pl.DataFrame,
    df_identity: pl.DataFrame,
) -> pl.DataFrame:
    """Merge transaction and identity DataFrames on TransactionID.

    Args:
        df_transaction: Transaction DataFrame.
        df_identity: Identity DataFrame.

    Returns:
        The merged Polars DataFrame.

    """
    return df_transaction.join(df_identity, on="TransactionID", how="left")


def column_types(df: pl.DataFrame, dirpath: Path) -> None:
    """Save the data types of each column in a DataFrame to a CSV file.

    Args:
        df: The Polars DataFrame.
        dirpath: Directory to save the CSV file.

    """
    dirpath.mkdir(parents=True, exist_ok=True)
    types = pl.DataFrame(
        {name: [str(dtype)] for name, dtype in zip(df.columns, df.dtypes, strict=True)},
    )
    types.write_csv(dirpath / "column_types.csv")


def null_profile(df: pl.DataFrame, dirpath: Path) -> None:
    """Calculate and save null counts and percentages to CSV files.

    Args:
        df: The Polars DataFrame.
        dirpath: Directory to save the CSV files.

    """
    dirpath.mkdir(parents=True, exist_ok=True)
    null_counts = df.null_count()
    null_percentages = null_counts / len(df)

    null_counts.write_csv(dirpath / "null_counts.csv")
    null_percentages.write_csv(dirpath / "null_percentages.csv")


def correlation_and_variance(df: pl.DataFrame, dirpath: Path) -> None:
    """Calculate and save the correlation matrix and variance of numeric features.

    Args:
        df: The Polars DataFrame.
        dirpath: Directory to save the CSV files.

    """
    dirpath.mkdir(parents=True, exist_ok=True)
    numeric_df = df.select(cs.numeric())

    correlation_matrix = numeric_df.corr()

    features = numeric_df.columns
    correlation_matrix = correlation_matrix.with_columns(
        pl.Series("feature", features),
    ).select(["feature", *features])

    correlation_matrix.write_csv(dirpath / "correlation_matrix.csv")

    variance_df = numeric_df.var()
    variance_df.write_csv(dirpath / "variance.csv")


def get_numeric_and_categorical_columns(
    df: pl.DataFrame,
    exclude_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Get the names of numeric and categorical columns in a DataFrame.

    Args:
        df: The Polars DataFrame.
        exclude_columns: List of column names to exclude.

    Returns:
        A tuple containing lists of numeric and categorical column names.

    """
    if exclude_columns is None:
        exclude_columns = []

    numeric_cols: list[str] = [
        col for col in df.select(cs.numeric()).columns if col not in exclude_columns
    ]
    categorical_cols: list[str] = [
        col
        for col in df.select(cs.string() | cs.categorical()).columns
        if col not in exclude_columns
    ]
    return numeric_cols, categorical_cols


def time_series_split(
    df: pl.DataFrame,
    target: str,
    timestamp: str,
    gap: int = 0,
    n_splits: int = 5,
) -> tuple[pdDataFrame, np.ndarray, TimeSeriesSplit]:
    """Perform a time-series split on a DataFrame.

    Args:
        df: The Polars DataFrame.
        target: The name of the target column.
        timestamp: The name of the column to sort by.
        gap: The number of samples to exclude between the training and test sets.
        n_splits: The number of splits.

    Returns:
        A tuple containing the features (X), the target (y), and the
        TimeSeriesSplit object.

    """
    df_sorted = df.sort(timestamp)
    x = df_sorted.select(pl.all().exclude(target)).to_pandas()
    y = df_sorted.select(target).to_pandas().to_numpy().ravel()
    tscv = TimeSeriesSplit(gap=gap, n_splits=n_splits)
    return x, y, tscv
