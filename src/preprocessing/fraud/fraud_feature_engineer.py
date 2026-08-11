from pathlib import Path
import json

import pandas as pd


# ============================================================
# FRAUD FEATURE ENGINEERING PIPELINE
# ============================================================

# src/preprocessing/fraud/fraud_feature_engineer.py
# parents[3] -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
    / "fraud_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fraud_features.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "fraud_feature_metadata.json"
)

TARGET_COLUMN = "SeriousDlqin2yrs"


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, required_columns):

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing)
        )


def safe_divide(numerator, denominator):

    denominator = denominator.replace(
        0,
        1,
    )

    return numerator / denominator


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FRAUD FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nChecking input dataset...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Fraud clean dataset not found:\n{INPUT_FILE}"
        )

    print(
        f"Input file: {INPUT_FILE}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input rows    : {len(df):,}"
    )

    print(
        f"Input columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        TARGET_COLUMN,
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

    require_columns(
        df,
        required_columns,
    )

    # --------------------------------------------------------
    # Copy
    # --------------------------------------------------------

    features = df.copy()

    # --------------------------------------------------------
    # Remove identifier
    # --------------------------------------------------------

    if "Unnamed: 0" in features.columns:

        features = features.drop(
            columns=["Unnamed: 0"]
        )

        print(
            "\nRemoved identifier: Unnamed: 0"
        )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_base_columns = [
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

    for column in numeric_base_columns:

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    print("\nCreating engineered features...")

    # --------------------------------------------------------
    # 1. Total delinquency events
    # --------------------------------------------------------

    features["TotalDelinquencyEvents"] = (
        features[
            "NumberOfTime30-59DaysPastDueNotWorse"
        ].fillna(0)
        +
        features[
            "NumberOfTime60-89DaysPastDueNotWorse"
        ].fillna(0)
        +
        features[
            "NumberOfTimes90DaysLate"
        ].fillna(0)
    )

    # --------------------------------------------------------
    # 2. Has delinquency
    # --------------------------------------------------------

    features["HasDelinquency"] = (
        features[
            "TotalDelinquencyEvents"
        ] > 0
    ).astype(int)

    # --------------------------------------------------------
    # 3. Severe delinquency
    # --------------------------------------------------------

    features["HasSevereDelinquency"] = (
        features[
            "NumberOfTimes90DaysLate"
        ] > 0
    ).astype(int)

    # --------------------------------------------------------
    # 4. High credit utilization
    # --------------------------------------------------------

    features["HighCreditUtilization"] = (
        features[
            "RevolvingUtilizationOfUnsecuredLines"
        ] >= 0.80
    ).astype(int)

    # --------------------------------------------------------
    # 5. High debt ratio
    # --------------------------------------------------------

    features["HighDebtRatio"] = (
        features[
            "DebtRatio"
        ] >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # 6. Income per dependent
    # --------------------------------------------------------

    dependents = (
        features[
            "NumberOfDependents"
        ]
        .fillna(0)
        .clip(lower=0)
    )

    features["IncomePerDependent"] = safe_divide(
        features[
            "MonthlyIncome"
        ].fillna(0),
        dependents + 1,
    )

    # --------------------------------------------------------
    # 7. Total credit lines
    # --------------------------------------------------------

    features["TotalCreditLines"] = (
        features[
            "NumberOfOpenCreditLinesAndLoans"
        ].fillna(0)
        +
        features[
            "NumberRealEstateLoansOrLines"
        ].fillna(0)
    )

    # --------------------------------------------------------
    # 8. Age band
    # --------------------------------------------------------

    features["AgeBand"] = pd.cut(
        features["age"],
        bins=[
            0,
            25,
            35,
            50,
            65,
            float("inf"),
        ],
        labels=[
            "18-25",
            "26-35",
            "36-50",
            "51-65",
            "66+",
        ],
        include_lowest=True,
    )

    features["AgeBand"] = (
        features["AgeBand"]
        .astype("string")
    )

    # --------------------------------------------------------
    # 9. Risk indicator
    # --------------------------------------------------------

    features["RiskIndicator"] = (
        (
            features[
                "TotalDelinquencyEvents"
            ] >= 2
        ).astype(int)
        +
        (
            features[
                "NumberOfTimes90DaysLate"
            ] > 0
        ).astype(int)
        +
        (
            features[
                "RevolvingUtilizationOfUnsecuredLines"
            ] >= 0.80
        ).astype(int)
        +
        (
            features[
                "DebtRatio"
            ] >= 0.50
        ).astype(int)
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if TARGET_COLUMN not in features.columns:

        raise ValueError(
            f"Target column missing: {TARGET_COLUMN}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    engineered_features = [
        "TotalDelinquencyEvents",
        "HasDelinquency",
        "HasSevereDelinquency",
        "HighCreditUtilization",
        "HighDebtRatio",
        "IncomePerDependent",
        "TotalCreditLines",
        "AgeBand",
        "RiskIndicator",
    ]

    print("\nFeature engineering summary")
    print("-" * 70)

    print(
        f"Final rows    : {len(features):,}"
    )

    print(
        f"Final columns : {len(features.columns)}"
    )

    print(
        f"New features  : {len(engineered_features)}"
    )

    print("\nEngineered features:")

    for feature in engineered_features:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution:")

    print(
        features[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_summary = {}

    for column, value in (
        features.isna().sum().items()
    ):

        if value > 0:

            missing_summary[column] = int(
                value
            )

    print(
        "\nMissing-value columns:"
    )

    if missing_summary:

        for column, value in missing_summary.items():

            print(
                f"  {column}: {value:,}"
            )

    else:

        print("  None")

    # --------------------------------------------------------
    # AgeBand distribution
    # --------------------------------------------------------

    print(
        "\nAgeBand distribution:"
    )

    print(
        features[
            "AgeBand"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # RiskIndicator distribution
    # --------------------------------------------------------

    print(
        "\nRiskIndicator distribution:"
    )

    print(
        features[
            "RiskIndicator"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # SAVE DATASET
    # ========================================================

    print(
        "\nSaving feature dataset..."
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "pipeline": (
            "Fraud Feature Engineering Pipeline"
        ),
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "target_column": TARGET_COLUMN,
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "output_rows": int(len(features)),
        "output_columns": int(len(features.columns)),
        "engineered_feature_count": int(
            len(engineered_features)
        ),
        "engineered_features": (
            engineered_features
        ),
        "identifier_removed": (
            "Unnamed: 0"
        ),
        "feature_logic": {
            "TotalDelinquencyEvents": (
                "30-59 + 60-89 + 90+ delinquency events"
            ),
            "HasDelinquency": (
                "TotalDelinquencyEvents > 0"
            ),
            "HasSevereDelinquency": (
                "NumberOfTimes90DaysLate > 0"
            ),
            "HighCreditUtilization": (
                "Utilization >= 0.80"
            ),
            "HighDebtRatio": (
                "DebtRatio >= 0.50"
            ),
            "IncomePerDependent": (
                "MonthlyIncome / (Dependents + 1)"
            ),
            "TotalCreditLines": (
                "Open credit lines + real estate loans"
            ),
            "AgeBand": (
                "18-25, 26-35, 36-50, 51-65, 66+"
            ),
            "RiskIndicator": (
                "Rule-based risk score from delinquency, "
                "utilization and debt ratio"
            ),
        },
        "target_distribution": {
            str(key): int(value)
            for key, value
            in features[
                TARGET_COLUMN
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "missing_values": missing_summary,
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "FRAUD FEATURE ENGINEERING COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nFeature dataset : {OUTPUT_FILE}"
    )

    print(
        f"Metadata        : {METADATA_FILE}"
    )

    print(
        f"Final rows      : {len(features):,}"
    )

    print(
        f"Final columns   : {len(features.columns)}"
    )

    print(
        "\nPipeline completed successfully."
    )


if __name__ == "__main__":
    main()