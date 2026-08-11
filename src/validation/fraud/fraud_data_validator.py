from pathlib import Path
import json

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fraud"
    / "cs-training.csv"
)

STAGING_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
)

STAGING_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_OUTPUT = (
    STAGING_DIR
    / "fraud_clean.csv"
)

PROFILE_OUTPUT = (
    STAGING_DIR
    / "fraud_profile.json"
)

QUALITY_OUTPUT = (
    STAGING_DIR
    / "fraud_quality_report.json"
)

TARGET_COLUMN = "SeriousDlqin2yrs"

ID_COLUMN = "Unnamed: 0"


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


def convert_inf_to_nan(df):
    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    if len(numeric_columns) > 0:
        df[numeric_columns] = df[numeric_columns].replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

    return df


# ============================================================
# MAIN VALIDATION PIPELINE
# ============================================================

def main():

    print_section("FRAUD DATA VALIDATION PIPELINE")

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    print("\nChecking required artifacts...")

    require_file(INPUT_FILE)

    print(f"Input file found:")
    print(INPUT_FILE)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading fraud dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    print_section("COLUMN VALIDATION")

    actual_columns = df.columns.tolist()

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    print(f"Expected columns : {len(EXPECTED_COLUMNS)}")
    print(f"Actual columns   : {len(actual_columns)}")

    if missing_columns:
        print("\nMissing columns:")
        for column in missing_columns:
            print(f"  - {column}")

        raise ValueError(
            "Dataset is missing required columns."
        )

    if unexpected_columns:
        print("\nUnexpected columns:")
        for column in unexpected_columns:
            print(f"  - {column}")

    print("\nRequired columns: PRESENT")

    # --------------------------------------------------------
    # Duplicate analysis
    # --------------------------------------------------------

    print_section("DUPLICATE ANALYSIS")

    duplicate_rows = int(df.duplicated().sum())

    print(f"Duplicate rows : {duplicate_rows:,}")

    # --------------------------------------------------------
    # Missing value analysis
    # --------------------------------------------------------

    print_section("MISSING VALUE ANALYSIS")

    missing_counts = df.isna().sum()

    missing_total = int(missing_counts.sum())

    missing_columns = (
        missing_counts[
            missing_counts > 0
        ]
        .sort_values(ascending=False)
    )

    print(f"Total missing values : {missing_total:,}")

    if len(missing_columns) == 0:
        print("Missing values      : NONE")
    else:
        print("\nColumns with missing values:")

        for column, count in missing_columns.items():
            percentage = (
                float(count) / len(df) * 100
            )

            print(
                f"{column:<45}"
                f"{count:>10,}"
                f" ({percentage:.2f}%)"
            )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    print_section("INFINITE VALUE ANALYSIS")

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    infinite_counts = {}

    for column in numeric_columns:

        count = int(
            df[column]
            .isin([float("inf"), float("-inf")])
            .sum()
        )

        if count > 0:
            infinite_counts[column] = count

    total_infinite = sum(
        infinite_counts.values()
    )

    print(
        f"Infinite values : {total_infinite:,}"
    )

    if infinite_counts:
        for column, count in infinite_counts.items():
            print(f"{column}: {count:,}")
    else:
        print("Infinite values : NONE")

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    print_section("TARGET VALIDATION")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET_COLUMN}"
        )

    target_dtype = str(
        df[TARGET_COLUMN].dtype
    )

    target_values = sorted(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    print(f"Target column : {TARGET_COLUMN}")
    print(f"Target dtype  : {target_dtype}")
    print(f"Target values : {target_values}")

    invalid_target_values = [
        value
        for value in target_values
        if value not in [0, 1]
    ]

    if invalid_target_values:
        raise ValueError(
            "Invalid target values detected: "
            f"{invalid_target_values}"
        )

    print("Target validation: PASSED")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print_section("TARGET DISTRIBUTION")

    target_counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    target_distribution = {}

    for value, count in target_counts.items():

        percentage = (
            float(count) / len(df) * 100
        )

        target_distribution[str(int(value))] = {
            "count": int(count),
            "percentage": round(
                percentage,
                4,
            ),
        }

        print(
            f"Class {int(value)}"
            f"{' ' * 5}"
            f": {int(count):,}"
            f" ({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    print_section("NUMERIC FEATURE VALIDATION")

    numeric_summary = {}

    for column in numeric_columns:

        series = df[column]

        numeric_summary[column] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique_values": int(
                series.nunique(dropna=True)
            ),
            "min": (
                float(series.min())
                if series.notna().any()
                else None
            ),
            "max": (
                float(series.max())
                if series.notna().any()
                else None
            ),
            "mean": (
                float(series.mean())
                if series.notna().any()
                else None
            ),
        }

    print(
        f"Numeric columns : {len(numeric_columns)}"
    )

    # --------------------------------------------------------
    # Basic business sanity checks
    # --------------------------------------------------------

    print_section("BUSINESS SANITY CHECKS")

    sanity_checks = {}

    # Age
    if "age" in df.columns:

        invalid_age = int(
            (
                (df["age"] < 18)
                | (df["age"] > 120)
            )
            .fillna(False)
            .sum()
        )

        sanity_checks["age_range"] = {
            "invalid_count": invalid_age,
            "status": (
                "PASS"
                if invalid_age == 0
                else "REVIEW"
            ),
        }

        print(
            f"Age range check       : "
            f"{'PASS' if invalid_age == 0 else 'REVIEW'}"
            f" ({invalid_age:,} invalid)"
        )

    # Revolving utilization
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:

        invalid_utilization = int(
            (
                df[
                    "RevolvingUtilizationOfUnsecuredLines"
                ] < 0
            )
            .fillna(False)
            .sum()
        )

        sanity_checks[
            "revolving_utilization"
        ] = {
            "invalid_count": invalid_utilization,
            "status": (
                "PASS"
                if invalid_utilization == 0
                else "REVIEW"
            ),
        }

        print(
            "Revolving utilization "
            f"check              : "
            f"{'PASS' if invalid_utilization == 0 else 'REVIEW'}"
            f" ({invalid_utilization:,} invalid)"
        )

    # Debt ratio
    if "DebtRatio" in df.columns:

        invalid_debt_ratio = int(
            (
                df["DebtRatio"] < 0
            )
            .fillna(False)
            .sum()
        )

        sanity_checks["debt_ratio"] = {
            "invalid_count": invalid_debt_ratio,
            "status": (
                "PASS"
                if invalid_debt_ratio == 0
                else "REVIEW"
            ),
        }

        print(
            f"Debt ratio check      : "
            f"{'PASS' if invalid_debt_ratio == 0 else 'REVIEW'}"
            f" ({invalid_debt_ratio:,} invalid)"
        )

    # Income
    if "MonthlyIncome" in df.columns:

        invalid_income = int(
            (
                df["MonthlyIncome"] < 0
            )
            .fillna(False)
            .sum()
        )

        sanity_checks["monthly_income"] = {
            "invalid_count": invalid_income,
            "status": (
                "PASS"
                if invalid_income == 0
                else "REVIEW"
            ),
        }

        print(
            f"Monthly income check  : "
            f"{'PASS' if invalid_income == 0 else 'REVIEW'}"
            f" ({invalid_income:,} invalid)"
        )

    # Dependents
    if "NumberOfDependents" in df.columns:

        invalid_dependents = int(
            (
                df["NumberOfDependents"] < 0
            )
            .fillna(False)
            .sum()
        )

        sanity_checks[
            "number_of_dependents"
        ] = {
            "invalid_count": invalid_dependents,
            "status": (
                "PASS"
                if invalid_dependents == 0
                else "REVIEW"
            ),
        }

        print(
            f"Dependents check      : "
            f"{'PASS' if invalid_dependents == 0 else 'REVIEW'}"
            f" ({invalid_dependents:,} invalid)"
        )

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    print_section("DATA CLEANING")

    clean_df = df.copy()

    # Remove identifier
    if ID_COLUMN in clean_df.columns:

        clean_df = clean_df.drop(
            columns=[ID_COLUMN]
        )

        print(
            f"Removed identifier column: "
            f"{ID_COLUMN}"
        )

    # Replace infinite numeric values
    clean_df = convert_inf_to_nan(
        clean_df
    )

    # --------------------------------------------------------
    # Save clean dataset
    # --------------------------------------------------------

    print("\nSaving clean dataset...")

    clean_df.to_csv(
        CLEAN_OUTPUT,
        index=False,
    )

    print(
        f"Clean dataset:"
    )
    print(CLEAN_OUTPUT)

    # --------------------------------------------------------
    # Profile metadata
    # --------------------------------------------------------

    profile = {
        "dataset": "credit-risk-fraud-source",
        "source_file": str(INPUT_FILE),
        "clean_file": str(CLEAN_OUTPUT),
        "rows": int(len(df)),
        "columns_before_cleaning": int(
            len(df.columns)
        ),
        "columns_after_cleaning": int(
            len(clean_df.columns)
        ),
        "target_column": TARGET_COLUMN,
        "target_distribution": target_distribution,
        "duplicate_rows": duplicate_rows,
        "missing_total": missing_total,
        "missing_columns": {
            str(column): int(count)
            for column, count
            in missing_counts.items()
            if count > 0
        },
        "numeric_columns": [
            str(column)
            for column in numeric_columns
        ],
        "numeric_summary": numeric_summary,
    }

    with PROFILE_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            profile,
            file,
            indent=4,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # Quality report
    # --------------------------------------------------------

    quality_report = {
        "dataset": "fraud",
        "source_file": str(INPUT_FILE),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "validation": {
            "required_columns_present": (
                len(missing_columns) == 0
            ),
            "target_valid": (
                len(invalid_target_values) == 0
            ),
            "duplicate_rows": duplicate_rows,
            "missing_values": missing_total,
            "infinite_values": int(
                total_infinite
            ),
        },
        "sanity_checks": sanity_checks,
        "status": "PASSED",
    }

    with QUALITY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            quality_report,
            file,
            indent=4,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print_section("VALIDATION COMPLETE")

    print(
        f"Clean dataset : {CLEAN_OUTPUT}"
    )

    print(
        f"Profile       : {PROFILE_OUTPUT}"
    )

    print(
        f"Quality report: {QUALITY_OUTPUT}"
    )

    print(
        f"\nFinal rows    : {len(clean_df):,}"
    )

    print(
        f"Final columns : {len(clean_df.columns)}"
    )

    print(
        "\nFraud data validation pipeline "
        "completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()