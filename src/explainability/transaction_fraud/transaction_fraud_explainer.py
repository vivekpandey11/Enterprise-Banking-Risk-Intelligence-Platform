from pathlib import Path
import json
import joblib
import pandas as pd


# ============================================================
# TRANSACTION FRAUD MODEL EXPLAINABILITY PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "transaction_fraud"
    / "transaction_fraud_best_model.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transaction_fraud"
    / "transaction_fraud_preprocessor.joblib"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "transaction_fraud"
    / "transaction_fraud_demo"
    / "transaction_fraud_demo_prediction.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "transaction_fraud"
    / "transaction_fraud_explainability"
)

FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "transaction_fraud_feature_importance.csv"
CUSTOMER_EXPLANATION_PATH = OUTPUT_DIR / "transaction_fraud_customer_explanation.json"
METADATA_PATH = OUTPUT_DIR / "transaction_fraud_explainability_metadata.json"


def main():

    print("\n# TRANSACTION FRAUD MODEL EXPLAINABILITY PIPELINE\n")

    print("Project root:")
    print(PROJECT_ROOT)

    print("\nChecking required artifacts...")

    required_files = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        PREDICTION_PATH,
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        print("\nERROR: Required artifacts missing:")
        for path in missing:
            print(path)
        raise FileNotFoundError("Required explainability artifacts are missing.")

    print("All required artifacts found.")

    print("\nModel:")
    print(MODEL_PATH)

    print("\nPreprocessor:")
    print(PREPROCESSOR_PATH)

    print("\nPrediction:")
    print(PREDICTION_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(MODEL_PATH)

    print("Model type:", type(model).__name__)

    # --------------------------------------------------------
    # Load preprocessor
    # --------------------------------------------------------

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Preprocessor type:", type(preprocessor).__name__)

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    print("\nLoading demo prediction...")

    with open(PREDICTION_PATH, "r", encoding="utf-8") as file:
        prediction_data = json.load(file)

    prediction = prediction_data["prediction"]
    transaction = prediction_data["transaction"]

    fraud_probability = float(prediction["fraud_probability"])
    risk_segment = prediction["risk_segment"]
    fraud_decision = prediction["fraud_decision"]
    business_threshold = float(prediction["business_threshold"])

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    print("\nExtracting feature names...")

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = list(transaction.keys())

    # --------------------------------------------------------
    # Global feature importance
    # --------------------------------------------------------

    print("\nExtracting global feature importance...")

    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        if len(importances) != len(feature_names):
            raise ValueError(
                f"Feature importance count ({len(importances)}) does not match "
                f"feature name count ({len(feature_names)})."
            )

        feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )

        feature_importance = feature_importance.sort_values(
            "importance",
            ascending=False
        ).reset_index(drop=True)

    elif hasattr(model, "coef_"):

        coefficients = model.coef_[0]

        if len(coefficients) != len(feature_names):
            raise ValueError(
                f"Coefficient count ({len(coefficients)}) does not match "
                f"feature name count ({len(feature_names)})."
            )

        feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": abs(coefficients),
            }
        )

        feature_importance = feature_importance.sort_values(
            "importance",
            ascending=False
        ).reset_index(drop=True)

    else:
        raise ValueError(
            "Model does not expose feature_importances_ or coef_."
        )

    print("Features explained:", len(feature_importance))

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Top features
    # --------------------------------------------------------

    print("\n## Top Model Features\n")

    top_features = feature_importance.head(10)

    for index, row in top_features.iterrows():
        print(
            f"{index + 1}. "
            f"{row['feature']:<55} "
            f"importance={row['importance']:.6f}"
        )

    # --------------------------------------------------------
    # Business explanation
    # --------------------------------------------------------

    print("\n## Business Explanation\n")

    amount = float(transaction.get("Amount", 0))
    time_value = float(transaction.get("Time", 0))

    explanation_points = []

    if amount >= 1000:
        explanation_points.append(
            f"Transaction amount is relatively high ({amount:,.2f})."
        )
    else:
        explanation_points.append(
            f"Transaction amount is relatively low ({amount:,.2f})."
        )

    if fraud_probability >= business_threshold:
        explanation_points.append(
            "Predicted fraud probability is above the business threshold."
        )
    else:
        explanation_points.append(
            "Predicted fraud probability is below the business threshold."
        )

    if fraud_probability >= 0.80:
        explanation_points.append(
            "Transaction requires high-priority fraud review."
        )
    elif fraud_probability >= 0.50:
        explanation_points.append(
            "Transaction requires additional fraud investigation."
        )
    else:
        explanation_points.append(
            "Transaction does not currently indicate high fraud risk."
        )

    explanation_points.append(
        f"Transaction time feature value is {time_value:.2f}."
    )

    explanation_points.append(
        f"Predicted fraud probability is {fraud_probability * 100:.2f}%."
    )

    explanation_points.append(
        f"Risk segment is classified as {risk_segment}."
    )

    for point in explanation_points:
        print("-", point)

    # --------------------------------------------------------
    # Customer / transaction explanation
    # --------------------------------------------------------

    print("\nGenerating transaction-level explanation...")

    transaction_explanation = {
        "transaction": transaction,
        "prediction": {
            "fraud_probability": fraud_probability,
            "fraud_probability_percent": round(
                fraud_probability * 100,
                4
            ),
            "risk_segment": risk_segment,
            "fraud_decision": fraud_decision,
            "business_threshold": business_threshold,
        },
        "top_model_features": [
            {
                "feature": str(row["feature"]),
                "importance": float(row["importance"]),
            }
            for _, row in top_features.iterrows()
        ],
        "business_explanation": explanation_points,
    }

    with open(
        CUSTOMER_EXPLANATION_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            transaction_explanation,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "pipeline": "transaction_fraud_model_explainability",
        "model_file": str(MODEL_PATH),
        "model_type": type(model).__name__,
        "preprocessor_file": str(PREPROCESSOR_PATH),
        "preprocessor_type": type(preprocessor).__name__,
        "prediction_file": str(PREDICTION_PATH),
        "feature_count": len(feature_importance),
        "explanation_method": (
            "feature_importances_"
            if hasattr(model, "feature_importances_")
            else "absolute_coefficients"
        ),
        "fraud_probability": fraud_probability,
        "risk_segment": risk_segment,
        "fraud_decision": fraud_decision,
        "business_threshold": business_threshold,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\nFeature importance:")
    print(FEATURE_IMPORTANCE_PATH)

    print("\nTransaction explanation:")
    print(CUSTOMER_EXPLANATION_PATH)

    print("\nMetadata:")
    print(METADATA_PATH)

    print(
        "\nTransaction fraud explainability pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()