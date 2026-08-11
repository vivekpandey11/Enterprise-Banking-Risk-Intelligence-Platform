from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# FRAUD RISK INFERENCE PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "fraud"
    / "fraud_best_model.joblib"
)

PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud"
    / "fraud_preprocessor.joblib"
)

THRESHOLD_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
    / "fraud_threshold_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fraud_demo_prediction.json"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except Exception:
        return {}


def find_threshold(
    data: dict,
    keys: list[str],
    default: float,
) -> float:

    normalized_keys = {
        key.lower().replace("-", "_").replace(" ", "_")
        for key in keys
    }

    def search(obj):

        if isinstance(obj, dict):

            for key, value in obj.items():

                normalized = (
                    str(key)
                    .lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                if normalized in normalized_keys:

                    try:
                        return float(value)
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                result = search(value)

                if result is not None:
                    return result

        elif isinstance(obj, list):

            for item in obj:

                result = search(item)

                if result is not None:
                    return result

        return None

    result = search(data)

    if result is None:
        return default

    return float(result)


def classify_risk(probability: float) -> str:

    if probability < 0.05:
        return "Low Risk"

    if probability < 0.15:
        return "Moderate Risk"

    if probability < 0.30:
        return "High Risk"

    return "Very High Risk"


def get_fraud_decision(
    probability: float,
    threshold: float,
) -> str:

    if probability >= threshold:
        return "FRAUD_REVIEW"

    return "LOW_RISK"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FRAUD RISK INFERENCE PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Check artifacts
    # --------------------------------------------------------

    print("\nChecking required artifacts...")

    require_file(
        MODEL_FILE,
        "Fraud best model",
    )

    require_file(
        PREPROCESSOR_FILE,
        "Fraud preprocessor",
    )

    require_file(
        THRESHOLD_METADATA_FILE,
        "Fraud threshold metadata",
    )

    print("All required artifacts found.")

    print("\nModel:")
    print(MODEL_FILE)

    print("\nPreprocessor:")
    print(PREPROCESSOR_FILE)

    print("\nThreshold metadata:")
    print(THRESHOLD_METADATA_FILE)

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type        : {type(model).__name__}"
    )

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        f"Preprocessor type : {type(preprocessor).__name__}"
    )

    threshold_metadata = load_json(
        THRESHOLD_METADATA_FILE
    )

    business_threshold = find_threshold(
        threshold_metadata,
        [
            "business_threshold",
            "business_oriented_threshold",
            "recall_threshold",
        ],
        0.20,
    )

    # --------------------------------------------------------
    # Demo customer
    # --------------------------------------------------------

    customer = {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 6000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
        "TotalDelinquencyEvents": 0,
        "HasDelinquency": 0,
        "HasSevereDelinquency": 0,
        "HighCreditUtilization": 0,
        "HighDebtRatio": 0,
        "IncomePerDependent": 3000.0,
        "TotalCreditLines": 9,
        "AgeBand": "36-50",
        "RiskIndicator": 0,
    }

    customer_df = pd.DataFrame(
        [customer]
    )

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    expected_features = [
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

    missing_features = [
        feature
        for feature in expected_features
        if feature not in customer_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing customer features:\n"
            + "\n".join(missing_features)
        )

    customer_df = customer_df[
        expected_features
    ]

    # --------------------------------------------------------
    # Transform input
    # --------------------------------------------------------

    print("\nTransforming customer input...")

    transformed_customer = preprocessor.transform(
        customer_df
    )

    print(
        f"Transformed feature count: "
        f"{transformed_customer.shape[1]}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nGenerating fraud probability...")

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Fraud model does not support predict_proba()."
        )

    probability = float(
        model.predict_proba(
            transformed_customer
        )[:, 1][0]
    )

    risk_segment = classify_risk(
        probability
    )

    decision = get_fraud_decision(
        probability,
        business_threshold,
    )

    fraud_prediction = int(
        probability >= business_threshold
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FRAUD PREDICTION RESULT")
    print("=" * 70)

    print(
        f"\nFraud Probability : {probability:.4f}"
    )

    print(
        f"Fraud Probability : {probability * 100:.2f}%"
    )

    print(
        f"Risk Segment      : {risk_segment}"
    )

    print(
        f"Business Threshold: {business_threshold:.2f}"
    )

    print(
        f"Fraud Decision    : {decision}"
    )

    print(
        f"Fraud Prediction  : {fraud_prediction}"
    )

    # --------------------------------------------------------
    # Save artifact
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_artifact = {
        "customer": customer,
        "prediction": {
            "fraud_probability": round(
                probability,
                6,
            ),
            "risk_segment": risk_segment,
            "fraud_decision": decision,
            "business_threshold": round(
                business_threshold,
                4,
            ),
            "fraud_prediction": fraud_prediction,
        },
        "model": {
            "file": str(MODEL_FILE),
            "type": type(model).__name__,
        },
        "preprocessor": {
            "file": str(PREPROCESSOR_FILE),
            "type": type(preprocessor).__name__,
        },
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            prediction_artifact,
            file,
            indent=2,
        )

    print("\nPrediction artifact:")
    print(OUTPUT_FILE)

    print(
        "\nFraud inference pipeline completed successfully."
    )


if __name__ == "__main__":
    main()