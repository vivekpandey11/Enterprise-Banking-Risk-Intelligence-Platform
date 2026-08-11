from pathlib import Path
import json

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "credit_risk"
    / "cs-training.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
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


def add_result(results, rule, status, failed_rows, message):
    results.append(
        {
            "rule": rule,
            "status": status,
            "failed_rows": int(failed_rows),
            "message": message,
        }
    )


def validate_dataset(df):
    results = []

    # 1. Required columns
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    add_result(
        results,
        "required_columns",
        "PASS" if not missing_columns else "FAIL",
        len(missing_columns),
        (
            "All required columns are present."
            if not missing_columns
            else f"Missing columns: {missing_columns}"
        ),
    )

    # 2. Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    add_result(
        results,
        "duplicate_rows",
        "PASS" if duplicate_rows == 0 else "FAIL",
        duplicate_rows,
        (
            "No duplicate rows found."
            if duplicate_rows == 0
            else f"{duplicate_rows} duplicate rows detected."
        ),
    )

    # 3. Target validation
    invalid_target = int(
        (~df["SeriousDlqin2yrs"].isin([0, 1])).sum()
    )

    add_result(
        results,
        "target_values",
        "PASS" if invalid_target == 0 else "FAIL",
        invalid_target,
        (
            "Target contains only 0 and 1."
            if invalid_target == 0
            else "Invalid target values detected."
        ),
    )

    # 4. Age validation
    # Business rule: lending customers must be between 18 and 100.
    invalid_age = int(
        ((df["age"] < 18) | (df["age"] > 100)).sum()
    )

    add_result(
        results,
        "age_range",
        "PASS" if invalid_age == 0 else "FAIL",
        invalid_age,
        (
            "Age values are within 18-100."
            if invalid_age == 0
            else f"{invalid_age} invalid age values detected."
        ),
    )

    # 5. Monthly income validation
    invalid_income = int(
        (df["MonthlyIncome"].dropna() < 0).sum()
    )

    add_result(
        results,
        "monthly_income",
        "PASS" if invalid_income == 0 else "FAIL",
        invalid_income,
        (
            "No negative monthly income values."
            if invalid_income == 0
            else f"{invalid_income} negative income values detected."
        ),
    )

    # 6. Debt ratio validation
    invalid_debt_ratio = int(
        (df["DebtRatio"] < 0).sum()
    )

    add_result(
        results,
        "debt_ratio",
        "PASS" if invalid_debt_ratio == 0 else "FAIL",
        invalid_debt_ratio,
        (
            "No negative debt ratios."
            if invalid_debt_ratio == 0
            else f"{invalid_debt_ratio} negative debt ratios detected."
        ),
    )

    # 7. Revolving utilization
    invalid_utilization = int(
        (df["RevolvingUtilizationOfUnsecuredLines"] < 0).sum()
    )

    add_result(
        results,
        "revolving_utilization",
        "PASS" if invalid_utilization == 0 else "FAIL",
        invalid_utilization,
        (
            "No negative utilization values."
            if invalid_utilization == 0
            else f"{invalid_utilization} negative utilization values."
        ),
    )

    # 8. Delinquency counts
    delinquency_columns = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]

    for column in delinquency_columns:
        invalid_values = int(
            (df[column] < 0).sum()
        )

        add_result(
            results,
            f"{column}_non_negative",
            "PASS" if invalid_values == 0 else "FAIL",
            invalid_values,
            (
                f"{column} contains no negative values."
                if invalid_values == 0
                else f"{invalid_values} negative values detected."
            ),
        )

    # 9. Number of dependents
    invalid_dependents = int(
        (df["NumberOfDependents"].dropna() < 0).sum()
    )

    add_result(
        results,
        "number_of_dependents",
        "PASS" if invalid_dependents == 0 else "FAIL",
        invalid_dependents,
        (
            "No negative dependent counts."
            if invalid_dependents == 0
            else f"{invalid_dependents} negative dependent counts."
        ),
    )

    # 10. Open credit lines
    invalid_credit_lines = int(
        (df["NumberOfOpenCreditLinesAndLoans"] < 0).sum()
    )

    add_result(
        results,
        "open_credit_lines",
        "PASS" if invalid_credit_lines == 0 else "FAIL",
        invalid_credit_lines,
        (
            "No negative credit-line counts."
            if invalid_credit_lines == 0
            else f"{invalid_credit_lines} negative credit-line counts."
        ),
    )

    failed_rules = sum(
        1
        for result in results
        if result["status"] == "FAIL"
    )

    overall_status = (
        "PASS"
        if failed_rules == 0
        else "FAIL"
    )

    return {
        "dataset": DATA_FILE.name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "overall_status": overall_status,
        "failed_rule_count": failed_rules,
        "rules": results,
    }


def main():
    print("=" * 70)
    print("CREDIT RISK DATA QUALITY VALIDATION")
    print("=" * 70)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    report = validate_dataset(df)

    output_file = (
        OUTPUT_DIR
        / "credit_risk_quality_report.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nOverall Status: {report['overall_status']}"
    )

    print(
        f"Failed Rules  : {report['failed_rule_count']}"
    )

    print("\nValidation Results")
    print("-" * 70)

    for result in report["rules"]:
        print(
            f"{result['status']:5} | "
            f"{result['rule']:45} | "
            f"Failed Rows: {result['failed_rows']}"
        )

    print("\nQuality report written to:")
    print(output_file)


if __name__ == "__main__":
    main()
