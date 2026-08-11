from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transaction_fraud"
    / "creditcard.csv"
)

STAGING_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "transaction_fraud"
)

CLEAN_FILE = STAGING_DIR / "transaction_fraud_clean.csv"
PROFILE_FILE = STAGING_DIR / "transaction_fraud_profile.json"
QUALITY_FILE = STAGING_DIR / "transaction_fraud_quality_report.json"


REQUIRED_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Class",
]


def ensure_paths():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def validate_columns(df):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    extra = [column for column in df.columns if column not in REQUIRED_COLUMNS]

    if extra:
        print(f"Warning: Extra columns found: {extra}")


def validate_target(df):
    unique_values = sorted(df["Class"].dropna().unique().tolist())

    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"Class column must contain only 0 and 1. "
            f"Found: {unique_values}"
        )


def main():
    print("=" * 70)
    print("TRANSACTION FRAUD DATA VALIDATION PIPELINE")
    print("=" * 70)

    ensure_paths()

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nChecking input dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Transaction fraud dataset not found:\n{INPUT_FILE}"
        )

    print(f"Input file: {INPUT_FILE}")

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows    : {len(df):,}")
    print(f"Input columns : {len(df.columns)}")

    print("\nValidating schema...")

    validate_columns(df)

    print("Schema validation: PASSED")

    print("\nChecking target...")

    validate_target(df)

    print("Target validation: PASSED")

    print("\nChecking missing values...")

    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())

    print(f"Total missing values: {total_missing:,}")

    print("\nChecking duplicates...")

    duplicate_count = int(df.duplicated().sum())

    print(f"Duplicate rows found: {duplicate_count:,}")

    print("\nTarget distribution:")

    target_distribution = (
        df["Class"]
        .value_counts()
        .sort_index()
    )

    print(target_distribution.to_string())

    fraud_count = int((df["Class"] == 1).sum())
    legitimate_count = int((df["Class"] == 0).sum())

    fraud_rate = (
        fraud_count / len(df)
        if len(df) > 0
        else 0
    )

    print(f"\nLegitimate transactions : {legitimate_count:,}")
    print(f"Fraud transactions      : {fraud_count:,}")
    print(f"Fraud rate              : {fraud_rate:.6%}")

    print("\nChecking numeric columns...")

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print(f"Numeric columns: {len(numeric_columns)}")

    print("\nChecking infinite values...")

    infinite_count = 0

    numeric_df = df[numeric_columns]

    for column in numeric_columns:
        infinite_count += int(
            (~numeric_df[column].map(pd.isna)
             & numeric_df[column].isin([float("inf"), float("-inf")]))
            .sum()
        )

    print(f"Infinite values: {infinite_count:,}")

    print("\nTransaction amount statistics:")

    amount_stats = df["Amount"].describe()

    print(amount_stats.to_string())

    print("\nHandling duplicate rows...")

    original_rows = len(df)

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    removed_duplicates = original_rows - len(df)

    print(
        f"Removed duplicate rows: {removed_duplicates:,}"
    )

    print(f"Final rows: {len(df):,}")

    final_fraud_count = int(
        (df["Class"] == 1).sum()
    )

    final_legitimate_count = int(
        (df["Class"] == 0).sum()
    )

    final_fraud_rate = (
        final_fraud_count / len(df)
        if len(df) > 0
        else 0
    )

    print("\nFinal target distribution:")

    print(
        pd.Series(
            {
                "Legitimate": final_legitimate_count,
                "Fraud": final_fraud_count,
            }
        ).to_string()
    )

    print(
        f"\nFinal fraud rate: {final_fraud_rate:.6%}"
    )

    print("\nSaving clean dataset...")

    df.to_csv(
        CLEAN_FILE,
        index=False
    )

    profile = {
        "dataset": "creditcard.csv",
        "dataset_type": "transaction_fraud",
        "source_file": str(INPUT_FILE),
        "original_rows": int(original_rows),
        "original_columns": int(len(REQUIRED_COLUMNS)),
        "duplicate_rows_found": int(duplicate_count),
        "duplicate_rows_removed": int(removed_duplicates),
        "final_rows": int(len(df)),
        "final_columns": int(len(df.columns)),
        "missing_values": int(total_missing),
        "infinite_values": int(infinite_count),
        "target_column": "Class",
        "target_values": [0, 1],
        "legitimate_transactions": int(final_legitimate_count),
        "fraud_transactions": int(final_fraud_count),
        "fraud_rate": float(final_fraud_rate),
        "numeric_feature_count": int(
            len(numeric_columns) - 1
        ),
        "amount_statistics": {
            "min": float(df["Amount"].min()),
            "max": float(df["Amount"].max()),
            "mean": float(df["Amount"].mean()),
            "median": float(df["Amount"].median()),
        },
    }

    quality_report = {
        "schema_valid": True,
        "target_valid": True,
        "missing_values": int(total_missing),
        "infinite_values": int(infinite_count),
        "duplicates_found": int(duplicate_count),
        "duplicates_removed": int(removed_duplicates),
        "final_rows": int(len(df)),
        "final_columns": int(len(df.columns)),
        "fraud_rate": float(final_fraud_rate),
        "quality_status": "PASSED",
    }

    with open(
        PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            profile,
            file,
            indent=2
        )

    with open(
        QUALITY_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            quality_report,
            file,
            indent=2
        )

    print("\n" + "=" * 70)
    print("VALIDATION OUTPUTS")
    print("=" * 70)

    print(f"\nClean dataset : {CLEAN_FILE}")
    print(f"Profile       : {PROFILE_FILE}")
    print(f"Quality report: {QUALITY_FILE}")

    print(f"\nFinal rows    : {len(df):,}")
    print(f"Final columns : {len(df.columns)}")

    print(
        "\nTransaction fraud data validation "
        "pipeline completed successfully."
    )


if __name__ == "__main__":
    main()