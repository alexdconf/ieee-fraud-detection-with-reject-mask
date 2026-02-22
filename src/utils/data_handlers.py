import os
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs


def load_csv_data(filepath: str) -> pl.DataFrame:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pl.read_csv(filepath)


def merge_transaction_and_identity(
    df_transaction: pl.DataFrame, df_identity: pl.DataFrame
) -> pl.DataFrame:
    result = df_transaction.join(df_identity, on="TransactionID", how="left")
    return result


def column_types(df: pl.DataFrame, dirpath: Path) -> None:
    types = pl.DataFrame(
        {
            "column_name": df.columns,
            "data_type": [str(t) for t in df.dtypes],
        }
    )
    types.write_csv(dirpath / "column_types.csv")


def null_profile(df: pl.DataFrame, dirpath: Path) -> None:
    null_counts = df.null_count()
    null_percentages = null_counts / len(df)

    null_counts.write_csv(dirpath / f"null_counts.csv")
    null_percentages.write_csv(dirpath / f"null_percentages.csv")


def correlation_and_variance(df: pl.DataFrame, target: str, dirpath: Path) -> None:
    # Select numeric features
    numeric_df = df.select(cs.numeric())

    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Add feature names as the first column for the pivot table format
    features = numeric_df.columns
    correlation_matrix = correlation_matrix.with_columns(
        pl.Series("feature", features)
    ).select(["feature"] + features)
    
    correlation_matrix.write_csv(dirpath / "correlation_matrix.csv")

    # Calculate variance
    variance_df = numeric_df.var()
    
    # Transpose to get (feature, variance) format
    variance_transposed = variance_df.transpose(
        include_header=True, header_name="feature", column_names=["variance"]
    )
    variance_transposed.write_csv(dirpath / "variance.csv")
