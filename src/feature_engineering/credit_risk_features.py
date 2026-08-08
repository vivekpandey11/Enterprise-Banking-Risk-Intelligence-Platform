from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
    / "credit_risk_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "credit_risk"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    # ---------------------------------------------------------
    # 1. Remove source-system row identifier
    # ---------------------------------------------------------

    if "Unnamed: 0" in result.columns:
        result = result.drop(columns=["Unnamed: 0"])

    # ---------------------------------------------------------
    # 2. Monthly income imputation
    # ---------------------------------------------------------

    result["MonthlyIncome"] = result["MonthlyIncome"].fillna(
        result["MonthlyIncome"].median()
    )

    # ---------------------------------------------------------
    # 3. Number of dependents imputation
    # ---------------------------------------------------------

    result["NumberOfDependents"] = result[
        "NumberOfDependents"
    ].fillna(0)

    # ---------------------------------------------------------
    # 4. Total delinquency events
    # ---------------------------------------------------------

    result["TotalDelinquencyEvents"] = (
        result["NumberOfTime30-59DaysPastDueNotWorse"]
        + result["NumberOfTimes90DaysLate"]
        + result["NumberOfTime60-89DaysPastDueNotWorse"]
    )

    # ---------------------------------------------------------
    # 5. Any delinquency flag
    # ---------------------------------------------------------

    result["HasDelinquency"] = (
        result["TotalDelinquencyEvents"] > 0
    ).astype(int)

    # ---------------------------------------------------------
    # 6. Severe delinquency flag
    # ---------------------------------------------------------

    result["HasSevereDelinquency"] = (
        result["NumberOfTimes90DaysLate"] > 0
    ).astype(int)

    # ---------------------------------------------------------
    # 7. Credit utilization risk bucket
    # ---------------------------------------------------------

    result["HighCreditUtilization"] = (
        result["RevolvingUtilizationOfUnsecuredLines"] > 0.80
    ).astype(int)

    # ---------------------------------------------------------
    # 8. Debt ratio risk flag
    # ---------------------------------------------------------

    result["HighDebtRatio"] = (
        result["DebtRatio"] > 0.50
    ).astype(int)

    # ---------------------------------------------------------
    # 9. Income per dependent
    # ---------------------------------------------------------

    result["IncomePerDependent"] = (
        result["MonthlyIncome"]
        / (result["NumberOfDependents"] + 1)
    )

    # ---------------------------------------------------------
    # 10. Total credit exposure indicators
    # ---------------------------------------------------------

    result["TotalCreditLines"] = (
        result["NumberOfOpenCreditLinesAndLoans"]
        + result["NumberRealEstateLoansOrLines"]
    )

    # ---------------------------------------------------------
    # 11. Age risk bands
    # ---------------------------------------------------------

    result["AgeBand"] = pd.cut(
        result["age"],
        bins=[17, 25, 35, 50, 65, 120],
        labels=[
            "18-25",
            "26-35",
            "36-50",
            "51-65",
            "66+",
        ],
        include_lowest=True,
    )

    # ---------------------------------------------------------
    # 12. Overall risk indicator
    # ---------------------------------------------------------

    result["RiskIndicator"] = (
        result["HasDelinquency"]
        + result["HasSevereDelinquency"]
        + result["HighCreditUtilization"]
        + result["HighDebtRatio"]
    )

    return result


def main():

    print("=" * 70)
    print("CREDIT RISK FEATURE ENGINEERING")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Clean staging dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput rows    : {len(df):,}")
    print(f"Input columns : {len(df.columns)}")

    features = create_features(df)

    output_file = (
        OUTPUT_DIR
        / "credit_risk_features.csv"
    )

    features.to_csv(
        output_file,
        index=False,
    )

    metadata = {
        "source_file": str(INPUT_FILE),
        "output_file": str(output_file),
        "input_rows": len(df),
        "output_rows": len(features),
        "input_columns": len(df.columns),
        "output_columns": len(features.columns),
        "new_features": [
            "TotalDelinquencyEvents",
            "HasDelinquency",
            "HasSevereDelinquency",
            "HighCreditUtilization",
            "HighDebtRatio",
            "IncomePerDependent",
            "TotalCreditLines",
            "AgeBand",
            "RiskIndicator",
        ],
    }

    metadata_file = (
        OUTPUT_DIR
        / "credit_risk_feature_metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nFEATURE ENGINEERING COMPLETE")
    print("-" * 70)
    print(f"Output rows    : {len(features):,}")
    print(f"Output columns : {len(features.columns)}")

    print("\nFeature dataset:")
    print(output_file)

    print("\nMetadata:")
    print(metadata_file)


if __name__ == "__main__":
    main()