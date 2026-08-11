from pathlib import Path
import json

import joblib
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "credit_risk"
    / "credit_risk_best_model.joblib"
)

PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "credit_risk"
    / "credit_risk_preprocessor.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CONFIGURATION
# =========================================================

BUSINESS_THRESHOLD = 0.18
BEST_F1_THRESHOLD = 0.24


# =========================================================
# REQUIRED RAW FEATURES
# =========================================================

REQUIRED_FEATURES = [
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


# =========================================================
# RISK CLASSIFICATION
# =========================================================

def classify_risk(probability):
    """
    Convert default probability into a business risk segment.
    """

    if probability < 0.18:
        return "Low Risk"

    if probability < 0.24:
        return "Moderate Risk"

    if probability < 0.50:
        return "High Risk"

    return "Very High Risk"


# =========================================================
# CREDIT DECISION
# =========================================================

def credit_decision(probability):
    """
    Business decision based on the selected threshold.
    """

    if probability >= BUSINESS_THRESHOLD:
        return "REVIEW"

    return "LOW_RISK"


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_input(customer):
    """
    Validate customer input before inference.
    """

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in customer
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            f"{missing_features}"
        )

    # -----------------------------------------------------
    # Basic business validation
    # -----------------------------------------------------

    age = customer["age"]

    if age < 18 or age > 120:
        raise ValueError(
            f"Invalid age: {age}. "
            "Expected range: 18-120."
        )

    monthly_income = customer[
        "MonthlyIncome"
    ]

    if pd.notna(monthly_income):
        if monthly_income < 0:
            raise ValueError(
                "MonthlyIncome cannot be negative."
            )

    debt_ratio = customer[
        "DebtRatio"
    ]

    if pd.notna(debt_ratio):
        if debt_ratio < 0:
            raise ValueError(
                "DebtRatio cannot be negative."
            )

    utilization = customer[
        "RevolvingUtilizationOfUnsecuredLines"
    ]

    if pd.notna(utilization):
        if utilization < 0:
            raise ValueError(
                "Revolving utilization cannot be negative."
            )


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_credit_risk(
    customer_data,
    model,
    preprocessor,
):
    """
    Generate credit risk prediction for one customer.
    """

    validate_input(
        customer_data
    )

    # -----------------------------------------------------
    # Convert input into DataFrame
    # -----------------------------------------------------

    customer_df = pd.DataFrame(
        [customer_data]
    )

    # -----------------------------------------------------
    # Keep exact feature order
    # -----------------------------------------------------

    customer_df = customer_df[
        REQUIRED_FEATURES
    ]

    # -----------------------------------------------------
    # Apply fitted preprocessing pipeline
    # -----------------------------------------------------

    processed_data = (
        preprocessor.transform(
            customer_df
        )
    )

    # -----------------------------------------------------
    # Generate probability
    # -----------------------------------------------------

    probability = float(
        model.predict_proba(
            processed_data
        )[0][1]
    )

    # -----------------------------------------------------
    # Risk classification
    # -----------------------------------------------------

    risk_segment = classify_risk(
        probability
    )

    # -----------------------------------------------------
    # Business decision
    # -----------------------------------------------------

    decision = credit_decision(
        probability
    )

    # -----------------------------------------------------
    # Default prediction
    # -----------------------------------------------------

    default_prediction = int(
        probability >= BUSINESS_THRESHOLD
    )

    return {
        "default_probability": round(
            probability,
            6,
        ),
        "risk_segment": risk_segment,
        "credit_decision": decision,
        "business_threshold": BUSINESS_THRESHOLD,
        "default_prediction": default_prediction,
    }


# =========================================================
# DEMO CUSTOMER
# =========================================================

def get_demo_customer():
    """
    Example customer used to test inference.
    """

    return {
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


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CREDIT RISK INFERENCE PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Validate artifacts
    # -----------------------------------------------------

    print("\nChecking model artifacts...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    if not PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:\n"
            f"{PREPROCESSOR_FILE}"
        )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type : "
        f"{type(model).__name__}"
    )

    # -----------------------------------------------------
    # Load preprocessor
    # -----------------------------------------------------

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        "Preprocessor loaded successfully."
    )

    # -----------------------------------------------------
    # Demo customer
    # -----------------------------------------------------

    customer = get_demo_customer()

    print("\nCustomer input")
    print("-" * 70)

    for key, value in customer.items():
        print(
            f"{key:45} : {value}"
        )

    # -----------------------------------------------------
    # Run prediction
    # -----------------------------------------------------

    print("\nGenerating credit risk prediction...")

    result = predict_credit_risk(
        customer_data=customer,
        model=model,
        preprocessor=preprocessor,
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CREDIT RISK ASSESSMENT")
    print("=" * 70)

    print(
        f"\nDefault Probability : "
        f"{result['default_probability']:.4f}"
    )

    print(
        f"Risk Segment        : "
        f"{result['risk_segment']}"
    )

    print(
        f"Business Threshold  : "
        f"{result['business_threshold']:.2f}"
    )

    print(
        f"Credit Decision     : "
        f"{result['credit_decision']}"
    )

    print(
        f"Default Prediction  : "
        f"{result['default_prediction']}"
    )

    # -----------------------------------------------------
    # Save prediction
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "credit_risk_demo_prediction.json"
    )

    output = {
        "customer": customer,
        "prediction": result,
        "model": {
            "file": str(
                MODEL_FILE
            ),
            "type": type(model).__name__,
        },
        "preprocessor": str(
            PREPROCESSOR_FILE
        ),
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print(
        "\nPrediction artifact:"
    )

    print(
        output_file
    )

    print(
        "\nInference pipeline completed successfully."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()