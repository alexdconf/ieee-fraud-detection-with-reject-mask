import sys
from datetime import datetime
from pathlib import Path

from utils.data_handlers import (
    load_csv_data,
    merge_transaction_and_identity,
    null_profile,
    correlation_and_variance,
    column_types
)
import constants


REPORTS_DIR = Path("reports")
TARGET = "isFraud"


def init(train_transactions_df, train_identity_df):
    df = merge_transaction_and_identity(train_transactions_df, train_identity_df)
    return df


def main():
    try:
        train_transactions_df = load_csv_data(constants.TRAIN_TRANSACTIONS)
        train_identity_df = load_csv_data(constants.TRAIN_IDENTITY)
    except FileNotFoundError as e:
        print(f"Error loading data from csv file: {e}")
        sys.exit(1)

    merge_df = init(train_transactions_df, train_identity_df)
    column_types(merge_df, REPORTS_DIR)
    null_profile(merge_df, REPORTS_DIR)
    correlation_and_variance(merge_df, TARGET, REPORTS_DIR)


if __name__ == "__main__":
    print("Program start...")
    begin = datetime.now()
    main()
    print("Program stop.")
    print(f"Elapsed runtime: {datetime.now() - begin}")
