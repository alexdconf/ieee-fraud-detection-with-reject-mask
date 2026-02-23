"""Constants for the IEEE Fraud Detection project."""

from pathlib import Path

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "ieee-fraud-detection"

TRAIN_TRANSACTIONS = DATASET_DIR / "train_transaction.csv"
TRAIN_IDENTITY = DATASET_DIR / "train_identity.csv"
TEST_TRANSACTIONS = DATASET_DIR / "test_transaction.csv"
TEST_IDENTITY = DATASET_DIR / "test_identity.csv"

REPORTS_DIR = Path("reports")
TARGET = "isFraud"
TIMESTAMP = "TransactionDT"
