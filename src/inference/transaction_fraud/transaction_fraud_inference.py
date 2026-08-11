from pathlib import Path
import json

import joblib
import pandas as pd


# ============================================================
# TRANSACTION FRAUD INFERENCE PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "transaction_fraud"
    / "transaction_fraud_best_model.joblib"
)

PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transaction_fraud"
    / "transaction_fraud_preprocessor.joblib"
)

THRESHOLD_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "transaction_fraud"
    / "transaction_fraud_threshold_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "transaction_fraud"
    / "transaction_fraud_demo"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "transaction_fraud_demo_prediction.json"
)


# ============================================================
# DEMO TRANSACTION
# ============================================================

DEMO_TRANSACTION = {
    "Time": 45000.0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62,
}


# ============================================================
# RISK SEGMENT
# ============================================================

def get_risk_segment(probability):

    if probability < 0.20:
        return "Low Risk"

    if probability < 0.50:
        return "Moderate Risk"

    if probability < 0.80:
        return "High Risk"

    return "Very High Risk"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRANSACTION FRAUD RISK INFERENCE PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # ========================================================
    # 1. Check artifacts
    # ========================================================

    print("\nChecking required artifacts...")

    required_files = [
        MODEL_FILE,
        PREPROCESSOR_FILE,
        THRESHOLD_METADATA_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required artifact not found:\n{file_path}"
            )

    print("All required artifacts found.")

    print("\nModel:")
    print(MODEL_FILE)

    print("\nPreprocessor:")
    print(PREPROCESSOR_FILE)

    print("\nThreshold metadata:")
    print(THRESHOLD_METADATA_FILE)

    # ========================================================
    # 2. Load model
    # ========================================================

    print("\nLoading model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type        : "
        f"{type(model).__name__}"
    )

    # ========================================================
    # 3. Load preprocessor
    # ========================================================

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        f"Preprocessor type : "
        f"{type(preprocessor).__name__}"
    )

    # ========================================================
    # 4. Load threshold metadata
    # ========================================================

    print("\nLoading threshold metadata...")

    with open(
        THRESHOLD_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    business_threshold = float(
        metadata["business_threshold"]
    )

    print(
        f"Business threshold: "
        f"{business_threshold:.2f}"
    )

    # ========================================================
    # 5. Prepare customer input
    # ========================================================

    print("\nTransforming transaction input...")

    transaction_df = pd.DataFrame(
        [DEMO_TRANSACTION]
    )

    expected_features = list(
        transaction_df.columns
    )

    transformed = preprocessor.transform(
        transaction_df
    )

    print(
        f"Transformed feature count: "
        f"{transformed.shape[1]}"
    )

    # ========================================================
    # 6. Generate probability
    # ========================================================

    print("\nGenerating fraud probability...")

    fraud_probability = float(
        model.predict_proba(
            transformed
        )[0, 1]
    )

    fraud_probability_pct = (
        fraud_probability * 100
    )

    # ========================================================
    # 7. Decision
    # ========================================================

    fraud_prediction = int(
        fraud_probability >= business_threshold
    )

    if fraud_prediction == 1:

        fraud_decision = "FRAUD_REVIEW"

    else:

        fraud_decision = "LOW_RISK"

    risk_segment = get_risk_segment(
        fraud_probability
    )

    # ========================================================
    # 8. Print result
    # ========================================================

    print("\n" + "=" * 70)
    print("TRANSACTION FRAUD RESULT")
    print("=" * 70)

    print(
        f"\nFraud Probability : "
        f"{fraud_probability:.4f}"
    )

    print(
        f"Fraud Probability : "
        f"{fraud_probability_pct:.2f}%"
    )

    print(
        f"Risk Segment      : "
        f"{risk_segment}"
    )

    print(
        f"Business Threshold: "
        f"{business_threshold:.2f}"
    )

    print(
        f"Fraud Decision    : "
        f"{fraud_decision}"
    )

    print(
        f"Fraud Prediction  : "
        f"{fraud_prediction}"
    )

    # ========================================================
    # 9. Save prediction artifact
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_artifact = {

        "transaction": DEMO_TRANSACTION,

        "prediction": {

            "fraud_probability":
                round(
                    fraud_probability,
                    6
                ),

            "fraud_probability_percent":
                round(
                    fraud_probability_pct,
                    4
                ),

            "risk_segment":
                risk_segment,

            "business_threshold":
                business_threshold,

            "fraud_decision":
                fraud_decision,

            "fraud_prediction":
                fraud_prediction,
        },

        "model": {

            "file":
                str(MODEL_FILE),

            "type":
                type(model).__name__,
        },

        "preprocessor": {

            "file":
                str(PREPROCESSOR_FILE),

            "type":
                type(preprocessor).__name__,
        },

        "threshold_metadata": {

            "file":
                str(THRESHOLD_METADATA_FILE),

            "business_threshold":
                business_threshold,

            "best_f1_threshold":
                metadata.get(
                    "best_f1_threshold"
                ),
        },
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            prediction_artifact,
            f,
            indent=2
        )

    print("\nPrediction artifact:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print(
        "Transaction fraud inference "
        "pipeline completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()