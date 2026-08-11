from pathlib import Path
import json
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "aml" / "aml_transactions_100k.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "staging" / "aml"

CLEAN_FILE = OUTPUT_DIR / "aml_transactions_clean.csv"
PROFILE_FILE = OUTPUT_DIR / "aml_transactions_profile.json"
QUALITY_FILE = OUTPUT_DIR / "aml_transactions_quality_report.json"


REQUIRED_COLUMNS = [
    "record_key",
    "timestamp",
    "from_bank",
    "from_account",
    "from_country",
    "to_bank",
    "to_account",
    "to_country",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]


def main():
    print("# AML DATA VALIDATION PIPELINE")
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking input dataset...")
    print(f"Input file: {INPUT_FILE}")
    print()

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    input_rows, input_columns = df.shape

    print(f"Input rows    : {input_rows:,}")
    print(f"Input columns : {input_columns}")
    print()

    # ---------------------------------------------------------
    # Schema validation
    # ---------------------------------------------------------

    print("Validating schema...")

    missing_required = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )

    print("Schema validation: PASSED")
    print()

    # ---------------------------------------------------------
    # Target validation
    # ---------------------------------------------------------

    print("Checking target...")

    target_values = set(df["is_laundering"].dropna().unique())

    if not target_values.issubset({True, False, 0, 1}):
        raise ValueError(
            f"Unexpected target values: {target_values}"
        )

    print("Target validation: PASSED")
    print()

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("Checking missing values...")

    missing_total = int(df.isna().sum().sum())

    print(f"Total missing values: {missing_total:,}")
    print()

    # ---------------------------------------------------------
    # Duplicate records
    # ---------------------------------------------------------

    print("Checking duplicates...")

    duplicate_rows = int(df.duplicated().sum())

    print(f"Duplicate rows found: {duplicate_rows:,}")
    print()

    # ---------------------------------------------------------
    # Numeric validation
    # ---------------------------------------------------------

    print("Checking numeric columns...")

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print(f"Numeric columns: {len(numeric_columns)}")
    print(numeric_columns)
    print()

    # ---------------------------------------------------------
    # Infinite values
    # ---------------------------------------------------------

    print("Checking infinite values...")

    if numeric_columns:
        infinite_values = int(
            np.isinf(df[numeric_columns].to_numpy()).sum()
        )
    else:
        infinite_values = 0

    print(f"Infinite values: {infinite_values:,}")
    print()

    # ---------------------------------------------------------
    # Target distribution
    # ---------------------------------------------------------

    print("AML target distribution:")

    target_distribution = df["is_laundering"].value_counts(
        dropna=False
    )

    print(target_distribution.to_string())
    print()

    laundering_count = int(
        (df["is_laundering"] == True).sum()
    )

    legitimate_count = int(
        (df["is_laundering"] == False).sum()
    )

    laundering_rate = (
        laundering_count / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(f"Legitimate transactions : {legitimate_count:,}")
    print(f"Laundering transactions : {laundering_count:,}")
    print(f"Laundering rate         : {laundering_rate:.6f}%")
    print()

    # ---------------------------------------------------------
    # Transaction amount statistics
    # ---------------------------------------------------------

    print("Transaction amount statistics:")

    amount_received_stats = df["amount_received"].describe()

    print(amount_received_stats.to_string())
    print()

    print("Paid amount statistics:")

    amount_paid_stats = df["amount_paid"].describe()

    print(amount_paid_stats.to_string())
    print()

    # ---------------------------------------------------------
    # Category statistics
    # ---------------------------------------------------------

    print("Unique banks:")
    print(f"From banks: {df['from_bank'].nunique():,}")
    print(f"To banks  : {df['to_bank'].nunique():,}")
    print()

    print("Unique accounts:")
    print(f"From accounts: {df['from_account'].nunique():,}")
    print(f"To accounts  : {df['to_account'].nunique():,}")
    print()

    print("Payment formats:")
    print(df["payment_format"].value_counts().to_string())
    print()

    print("Currencies:")
    print(
        "Receiving currencies:",
        df["receiving_currency"].nunique()
    )
    print(
        "Payment currencies:",
        df["payment_currency"].nunique()
    )
    print()

    # ---------------------------------------------------------
    # Duplicate handling
    # ---------------------------------------------------------

    print("Handling duplicate rows...")

    before = len(df)

    df_clean = df.drop_duplicates().copy()

    removed_duplicates = before - len(df_clean)

    print(f"Removed duplicate rows: {removed_duplicates:,}")
    print(f"Final rows: {len(df_clean):,}")
    print()

    # ---------------------------------------------------------
    # Final target distribution
    # ---------------------------------------------------------

    final_laundering = int(
        (df_clean["is_laundering"] == True).sum()
    )

    final_legitimate = int(
        (df_clean["is_laundering"] == False).sum()
    )

    final_rate = (
        final_laundering / len(df_clean) * 100
        if len(df_clean) > 0
        else 0
    )

    print("Final target distribution:")

    print(
        f"Legitimate    {final_legitimate:,}"
    )

    print(
        f"Laundering    {final_laundering:,}"
    )

    print(
        f"Final laundering rate: {final_rate:.6f}%"
    )

    print()

    # ---------------------------------------------------------
    # Save clean dataset
    # ---------------------------------------------------------

    print("Saving clean dataset...")

    df_clean.to_csv(
        CLEAN_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Profile
    # ---------------------------------------------------------

    profile = {
        "dataset": "AML Transactions",
        "source_file": str(INPUT_FILE),
        "input_rows": int(input_rows),
        "input_columns": int(input_columns),
        "final_rows": int(len(df_clean)),
        "final_columns": int(len(df_clean.columns)),
        "duplicate_rows": int(duplicate_rows),
        "removed_duplicates": int(removed_duplicates),
        "missing_values": int(missing_total),
        "infinite_values": int(infinite_values),
        "laundering_transactions": int(final_laundering),
        "legitimate_transactions": int(final_legitimate),
        "laundering_rate_percent": float(final_rate),
        "numeric_columns": numeric_columns,
        "columns": df_clean.columns.tolist(),
    }

    with open(
        PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            profile,
            f,
            indent=2
        )

    # ---------------------------------------------------------
    # Quality report
    # ---------------------------------------------------------

    quality_report = {
        "status": "PASSED",
        "schema_validation": "PASSED",
        "target_validation": "PASSED",
        "input_rows": int(input_rows),
        "final_rows": int(len(df_clean)),
        "input_columns": int(input_columns),
        "final_columns": int(len(df_clean.columns)),
        "missing_values": int(missing_total),
        "duplicate_rows_found": int(duplicate_rows),
        "duplicates_removed": int(removed_duplicates),
        "infinite_values": int(infinite_values),
        "laundering_transactions": int(final_laundering),
        "legitimate_transactions": int(final_legitimate),
        "laundering_rate_percent": float(final_rate),
    }

    with open(
        QUALITY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            quality_report,
            f,
            indent=2
        )

    print()
    print("Output artifacts:")
    print()
    print(f"Clean dataset : {CLEAN_FILE}")
    print(f"Profile       : {PROFILE_FILE}")
    print(f"Quality report: {QUALITY_FILE}")
    print()
    print(f"Final rows    : {len(df_clean):,}")
    print(f"Final columns : {len(df_clean.columns)}")
    print()
    print("AML data validation pipeline completed successfully.")


if __name__ == "__main__":
    main()