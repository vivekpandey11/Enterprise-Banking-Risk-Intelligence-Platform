from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

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

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
    / "credit_risk_demo_prediction.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
    / "credit_risk_explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONSTANTS
# ============================================================

TARGET_COLUMN = "SeriousDlqin2yrs"

BUSINESS_THRESHOLD = 0.18

RISK_THRESHOLDS = {
    "Low Risk": 0.05,
    "Moderate Risk": 0.10,
    "High Risk": 0.18,
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def require_file(path: Path, description: str) -> None:
    """Verify that a required artifact exists."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} is not a file:\n{path}"
        )


def load_json(path: Path) -> dict:
    """Load JSON file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(path: Path, data: dict) -> None:
    """Save JSON file."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            default=str,
        )


def determine_risk_segment(probability: float) -> str:
    """Convert default probability into business risk segment."""

    if probability < RISK_THRESHOLDS["Low Risk"]:
        return "Low Risk"

    if probability < RISK_THRESHOLDS["Moderate Risk"]:
        return "Moderate Risk"

    if probability < RISK_THRESHOLDS["High Risk"]:
        return "High Risk"

    return "Very High Risk"


def determine_credit_decision(probability: float) -> str:
    """Convert probability into business credit decision."""

    if probability >= BUSINESS_THRESHOLD:
        return "HIGH_RISK"

    return "LOW_RISK"


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    model,
    preprocessor,
) -> pd.DataFrame:
    """
    Extract model feature importance.

    Supports tree models with feature_importances_
    and linear models with coef_.
    """

    try:
        feature_names = (
            preprocessor
            .get_feature_names_out()
        )
    except Exception:
        feature_names = None

    importance_values = None

    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(
            model.feature_importances_,
            dtype=float,
        )

    elif hasattr(model, "coef_"):
        coefficients = np.asarray(
            model.coef_,
            dtype=float,
        )

        if coefficients.ndim == 2:
            coefficients = coefficients[0]

        importance_values = np.abs(
            coefficients
        )

    if importance_values is None:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "absolute_importance",
                "rank",
            ]
        )

    if feature_names is None:
        feature_names = [
            f"feature_{index}"
            for index in range(
                len(importance_values)
            )
        ]

    feature_names = list(feature_names)

    if len(feature_names) != len(
        importance_values
    ):
        feature_names = [
            f"feature_{index}"
            for index in range(
                len(importance_values)
            )
        ]

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
            "absolute_importance": np.abs(
                importance_values
            ),
        }
    )

    result = result.sort_values(
        "absolute_importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result["rank"] = (
        result.index + 1
    )

    return result


# ============================================================
# CUSTOMER FEATURE CONTRIBUTION
# ============================================================

def explain_customer_features(
    customer: dict,
    feature_importance: pd.DataFrame,
) -> list:
    """
    Build a business-readable explanation by combining
    customer's actual values with global model importance.
    """

    explanations = []

    if feature_importance.empty:
        return explanations

    for _, row in feature_importance.head(10).iterrows():

        feature_name = str(
            row["feature"]
        )

        clean_name = feature_name

        if clean_name.startswith(
            "numeric__"
        ):
            clean_name = clean_name.replace(
                "numeric__",
                "",
                1,
            )

        elif clean_name.startswith(
            "categorical__"
        ):
            clean_name = clean_name.replace(
                "categorical__",
                "",
                1,
            )

        value = customer.get(
            clean_name,
            None,
        )

        explanations.append(
            {
                "feature": clean_name,
                "model_feature": feature_name,
                "customer_value": value,
                "importance": round(
                    float(
                        row["absolute_importance"]
                    ),
                    6,
                ),
                "rank": int(
                    row["rank"]
                ),
            }
        )

    return explanations


# ============================================================
# BUSINESS EXPLANATION
# ============================================================

def generate_business_explanation(
    customer: dict,
    probability: float,
    risk_segment: str,
    decision: str,
) -> list:

    reasons = []

    utilization = float(
        customer.get(
            "RevolvingUtilizationOfUnsecuredLines",
            0,
        )
        or 0
    )

    debt_ratio = float(
        customer.get(
            "DebtRatio",
            0,
        )
        or 0
    )

    income = float(
        customer.get(
            "MonthlyIncome",
            0,
        )
        or 0
    )

    delinquency_events = int(
        customer.get(
            "TotalDelinquencyEvents",
            0,
        )
        or 0
    )

    severe_delinquency = int(
        customer.get(
            "HasSevereDelinquency",
            0,
        )
        or 0
    )

    high_utilization = int(
        customer.get(
            "HighCreditUtilization",
            0,
        )
        or 0
    )

    high_debt_ratio = int(
        customer.get(
            "HighDebtRatio",
            0,
        )
        or 0
    )

    if delinquency_events > 0:
        reasons.append(
            f"Customer has {delinquency_events} "
            "delinquency event(s)."
        )

    else:
        reasons.append(
            "No delinquency events are present."
        )

    if severe_delinquency > 0:
        reasons.append(
            "Severe delinquency indicator is present."
        )

    if high_utilization > 0:
        reasons.append(
            "Credit utilization is flagged as high."
        )

    elif utilization >= 0.70:
        reasons.append(
            f"Credit utilization is relatively high "
            f"({utilization:.2f})."
        )

    else:
        reasons.append(
            f"Credit utilization is moderate/low "
            f"({utilization:.2f})."
        )

    if high_debt_ratio > 0:
        reasons.append(
            "Debt ratio is flagged as high."
        )

    elif debt_ratio > 1:
        reasons.append(
            f"Debt ratio is elevated ({debt_ratio:.2f})."
        )

    else:
        reasons.append(
            f"Debt ratio is within the observed range "
            f"({debt_ratio:.2f})."
        )

    if income > 0:
        reasons.append(
            f"Monthly income is approximately "
            f"{income:,.2f}."
        )

    reasons.append(
        f"Predicted default probability is "
        f"{probability:.2%}."
    )

    reasons.append(
        f"Risk segment is classified as "
        f"{risk_segment}."
    )

    reasons.append(
        f"Business credit decision is "
        f"{decision}."
    )

    return reasons


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CREDIT RISK MODEL EXPLAINABILITY PIPELINE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check artifacts
    # --------------------------------------------------------

    print("\nChecking required artifacts...")

    require_file(
        MODEL_FILE,
        "Best model",
    )

    require_file(
        PREPROCESSOR_FILE,
        "Credit risk preprocessor",
    )

    require_file(
        PREDICTION_FILE,
        "Demo prediction",
    )

    print(
        "\nAll required artifacts found."
    )

    print(
        f"\nModel:\n{MODEL_FILE}"
    )

    print(
        f"\nPreprocessor:\n{PREPROCESSOR_FILE}"
    )

    print(
        f"\nPrediction:\n{PREDICTION_FILE}"
    )

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type: {type(model).__name__}"
    )

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        f"Preprocessor type: "
        f"{type(preprocessor).__name__}"
    )

    print("\nLoading demo prediction...")

    prediction_data = load_json(
        PREDICTION_FILE
    )

    customer = prediction_data.get(
        "customer",
        {},
    )

    original_prediction = prediction_data.get(
        "prediction",
        {},
    )

    # --------------------------------------------------------
    # Extract probability
    # --------------------------------------------------------

    probability = float(
        original_prediction.get(
            "default_probability",
            0,
        )
    )

    risk_segment = determine_risk_segment(
        probability
    )

    credit_decision = determine_credit_decision(
        probability
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print(
        "\nExtracting global feature importance..."
    )

    feature_importance = (
        extract_feature_importance(
            model,
            preprocessor,
        )
    )

    print(
        f"Features explained: "
        f"{len(feature_importance):,}"
    )

    # --------------------------------------------------------
    # Save feature importance
    # --------------------------------------------------------

    feature_importance_file = (
        OUTPUT_DIR
        / "credit_risk_feature_importance.csv"
    )

    feature_importance.to_csv(
        feature_importance_file,
        index=False,
    )

    # --------------------------------------------------------
    # Customer-level explanation
    # --------------------------------------------------------

    print(
        "\nGenerating customer-level explanation..."
    )

    customer_features = (
        explain_customer_features(
            customer,
            feature_importance,
        )
    )

    # --------------------------------------------------------
    # Business explanation
    # --------------------------------------------------------

    business_reasons = (
        generate_business_explanation(
            customer=customer,
            probability=probability,
            risk_segment=risk_segment,
            decision=credit_decision,
        )
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CUSTOMER RISK EXPLANATION")
    print("=" * 70)

    print(
        f"\nDefault Probability : "
        f"{probability:.4f}"
    )

    print(
        f"Risk Segment        : "
        f"{risk_segment}"
    )

    print(
        f"Business Threshold  : "
        f"{BUSINESS_THRESHOLD:.2f}"
    )

    print(
        f"Credit Decision     : "
        f"{credit_decision}"
    )

    print(
        "\nTop Model Features"
    )

    print("-" * 70)

    for item in customer_features[:10]:

        print(
            f"{item['rank']:>2}. "
            f"{item['feature']:<45} "
            f"importance="
            f"{item['importance']:.6f}"
        )

    print(
        "\nBusiness Explanation"
    )

    print("-" * 70)

    for reason in business_reasons:
        print(
            f"- {reason}"
        )

    # --------------------------------------------------------
    # Final explanation artifact
    # --------------------------------------------------------

    explanation = {
        "customer": customer,
        "prediction": {
            "default_probability": round(
                probability,
                6,
            ),
            "risk_segment": risk_segment,
            "business_threshold": BUSINESS_THRESHOLD,
            "credit_decision": credit_decision,
            "default_prediction": int(
                probability >= BUSINESS_THRESHOLD
            ),
        },
        "model": {
            "file": str(
                MODEL_FILE
            ),
            "type": type(model).__name__,
        },
        "preprocessor": {
            "file": str(
                PREPROCESSOR_FILE
            ),
            "type": type(
                preprocessor
            ).__name__,
        },
        "global_feature_importance": (
            feature_importance[
                [
                    "feature",
                    "importance",
                    "rank",
                ]
            ]
            .head(20)
            .to_dict(
                orient="records"
            )
        ),
        "customer_feature_explanation": (
            customer_features
        ),
        "business_explanation": (
            business_reasons
        ),
    }

    explanation_file = (
        OUTPUT_DIR
        / "credit_risk_customer_explanation.json"
    )

    save_json(
        explanation_file,
        explanation,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "pipeline": (
            "credit_risk_model_explainability"
        ),
        "model_type": type(model).__name__,
        "model_file": str(MODEL_FILE),
        "preprocessor_file": str(
            PREPROCESSOR_FILE
        ),
        "prediction_file": str(
            PREDICTION_FILE
        ),
        "business_threshold": (
            BUSINESS_THRESHOLD
        ),
        "default_probability": (
            round(
                probability,
                6,
            )
        ),
        "risk_segment": risk_segment,
        "credit_decision": credit_decision,
        "feature_count": int(
            len(feature_importance)
        ),
        "top_features_exported": 20,
    }

    metadata_file = (
        OUTPUT_DIR
        / "credit_risk_explainability_metadata.json"
    )

    save_json(
        metadata_file,
        metadata,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "EXPLAINABILITY PIPELINE COMPLETED"
    )
    print("=" * 70)

    print(
        f"\nFeature importance:"
        f"\n{feature_importance_file}"
    )

    print(
        f"\nCustomer explanation:"
        f"\n{explanation_file}"
    )

    print(
        f"\nMetadata:"
        f"\n{metadata_file}"
    )

    print(
        "\nPipeline completed successfully."
    )


if __name__ == "__main__":
    main()