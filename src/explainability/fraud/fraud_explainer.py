from pathlib import Path
import json
import joblib
import pandas as pd


PROJECT_ROOT = Path(r"D:\Enterprise-Banking-Risk-Intelligence-Platform")

MODEL_PATH = PROJECT_ROOT / "models" / "fraud" / "fraud_best_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "data" / "processed" / "fraud" / "fraud_preprocessor.joblib"
PREDICTION_PATH = PROJECT_ROOT / "data" / "staging" / "fraud" / "fraud_demo_prediction.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "staging" / "fraud" / "fraud_explainability"

FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "fraud_feature_importance.csv"
CUSTOMER_EXPLANATION_PATH = OUTPUT_DIR / "fraud_customer_explanation.json"
METADATA_PATH = OUTPUT_DIR / "fraud_explainability_metadata.json"


def require_file(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    print("=" * 70)
    print("FRAUD MODEL EXPLAINABILITY PIPELINE")
    print("=" * 70)

    print("\nChecking required artifacts...")

    require_file(MODEL_PATH, "Best fraud model")
    require_file(PREPROCESSOR_PATH, "Fraud preprocessor")
    require_file(PREDICTION_PATH, "Fraud prediction")

    print("\nAll required artifacts found.")

    print("\nModel:")
    print(MODEL_PATH)

    print("\nPreprocessor:")
    print(PREPROCESSOR_PATH)

    print("\nPrediction:")
    print(PREDICTION_PATH)

    print("\nLoading model...")

    model = joblib.load(MODEL_PATH)

    print("Model type:", type(model).__name__)

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print(
        "Preprocessor type:",
        type(preprocessor).__name__
    )

    print("\nLoading demo prediction...")

    prediction_data = load_json(PREDICTION_PATH)

    customer = prediction_data.get("customer", {})
    prediction = prediction_data.get("prediction", {})

    fraud_probability = float(
        prediction.get("fraud_probability", 0.0)
    )

    risk_segment = prediction.get(
        "risk_segment",
        "Unknown"
    )

    fraud_decision = prediction.get(
        "fraud_decision",
        "UNKNOWN"
    )

    business_threshold = float(
        prediction.get(
            "business_threshold",
            0.20
        )
    )

    print("\nExtracting global feature importance...")

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "Loaded fraud model does not contain "
            "feature_importances_."
        )

    importances = model.feature_importances_

    try:
        feature_names = list(
            preprocessor.get_feature_names_out()
        )
    except Exception:
        feature_names = []

    if len(feature_names) != len(importances):
        feature_names = [
            f"feature_{index + 1}"
            for index in range(len(importances))
        ]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df["rank"] = (
        importance_df.index + 1
    )

    importance_df = importance_df[
        [
            "rank",
            "feature",
            "importance"
        ]
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False
    )

    print(
        "Features explained:",
        len(importance_df)
    )

    print("\nGenerating customer-level explanation...")

    print(
        "\nFraud Probability :",
        f"{fraud_probability:.4f}"
    )

    print(
        "Fraud Probability :",
        f"{fraud_probability * 100:.2f}%"
    )

    print(
        "Risk Segment      :",
        risk_segment
    )

    print(
        "Business Threshold:",
        f"{business_threshold:.2f}"
    )

    print(
        "Fraud Decision     :",
        fraud_decision
    )

    print("\n## Top Model Features")

    top_features = importance_df.head(10)

    for _, row in top_features.iterrows():

        print(
            f"{int(row['rank']):2d}. "
            f"{str(row['feature']):45s} "
            f"importance={float(row['importance']):.6f}"
        )

    utilization = float(
        customer.get(
            "RevolvingUtilizationOfUnsecuredLines",
            0.0
        )
    )

    debt_ratio = float(
        customer.get(
            "DebtRatio",
            0.0
        )
    )

    monthly_income = float(
        customer.get(
            "MonthlyIncome",
            0.0
        )
    )

    delinquency_events = int(
        customer.get(
            "TotalDelinquencyEvents",
            0
        )
    )

    business_explanation = []

    if delinquency_events == 0:
        business_explanation.append(
            "No delinquency events are present."
        )
    else:
        business_explanation.append(
            f"{delinquency_events} delinquency event(s) are present."
        )

    if utilization <= 0.30:
        business_explanation.append(
            f"Credit utilization is low ({utilization:.2f})."
        )
    elif utilization <= 0.70:
        business_explanation.append(
            f"Credit utilization is moderate ({utilization:.2f})."
        )
    else:
        business_explanation.append(
            f"Credit utilization is high ({utilization:.2f})."
        )

    if debt_ratio <= 0.50:
        business_explanation.append(
            f"Debt ratio is relatively low ({debt_ratio:.2f})."
        )
    elif debt_ratio <= 1.00:
        business_explanation.append(
            f"Debt ratio is within the observed range ({debt_ratio:.2f})."
        )
    else:
        business_explanation.append(
            f"Debt ratio is elevated ({debt_ratio:.2f})."
        )

    business_explanation.append(
        f"Monthly income is approximately {monthly_income:,.2f}."
    )

    business_explanation.append(
        f"Predicted fraud probability is "
        f"{fraud_probability * 100:.2f}%."
    )

    business_explanation.append(
        f"Risk segment is classified as {risk_segment}."
    )

    print("\n## Business Explanation")

    for explanation in business_explanation:
        print("-", explanation)

    customer_explanation = {
        "customer": customer,
        "prediction": {
            "fraud_probability": round(
                fraud_probability,
                6
            ),
            "fraud_probability_percent": round(
                fraud_probability * 100,
                4
            ),
            "risk_segment": risk_segment,
            "fraud_decision": fraud_decision,
            "business_threshold": business_threshold
        },
        "business_explanation": business_explanation,
        "top_model_features": [
            {
                "rank": int(row["rank"]),
                "feature": str(row["feature"]),
                "importance": round(
                    float(row["importance"]),
                    6
                )
            }
            for _, row in top_features.iterrows()
        ]
    }

    with open(
        CUSTOMER_EXPLANATION_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            customer_explanation,
            file,
            indent=2,
            ensure_ascii=False
        )

    metadata = {
        "pipeline": "Fraud Model Explainability Pipeline",
        "project_root": str(PROJECT_ROOT),
        "model": {
            "file": str(MODEL_PATH),
            "type": type(model).__name__
        },
        "preprocessor": {
            "file": str(PREPROCESSOR_PATH),
            "type": type(preprocessor).__name__
        },
        "prediction_artifact": str(PREDICTION_PATH),
        "fraud_probability": round(
            fraud_probability,
            6
        ),
        "risk_segment": risk_segment,
        "fraud_decision": fraud_decision,
        "business_threshold": business_threshold,
        "feature_count": int(
            len(importance_df)
        ),
        "output_files": {
            "feature_importance": str(
                FEATURE_IMPORTANCE_PATH
            ),
            "customer_explanation": str(
                CUSTOMER_EXPLANATION_PATH
            ),
            "metadata": str(
                METADATA_PATH
            )
        }
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
            ensure_ascii=False
        )

    print("\nFeature importance:")
    print(FEATURE_IMPORTANCE_PATH)

    print("\nCustomer explanation:")
    print(CUSTOMER_EXPLANATION_PATH)

    print("\nMetadata:")
    print(METADATA_PATH)

    print(
        "\nFraud explainability pipeline completed successfully."
    )


if __name__ == "__main__":
    main()
