from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


# Necessary to generate "Timestamp_TransactionsDT_plot.jpg"
def timestamp_plot(df: pl.DataFrame, dirpath: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(df["TransactionDT"], range(len(df["TransactionDT"])), marker='o', linestyle='-', color='b')
    plt.title('Timestamp Trend')
    plt.xlabel('Count')
    plt.ylabel('Timestamp Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(dirpath / 'Timestamp_TransactionsDT_plot.jpg', format='jpg', dpi=300)
    plt.close()

# Necessary to generate "Target_isFraud_class_balance.jpg"
def target_class_balance(df: pl.DataFrame, dirpath: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.bar(df["isFraud"], len(df["isFraud"]))
    plt.title('Class Balance')
    plt.xlabel('isFraud Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(dirpath / 'Timestamp_TransactionsDT_plot.jpg', format='jpg', dpi=300)
    plt.close()
