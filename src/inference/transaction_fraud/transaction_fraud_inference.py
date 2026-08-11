from pathlib import Path
import json
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "transaction_fraud"
MODEL_DIR = PROJECT_ROOT / "models" / "transaction_fraud"
OUTPUT_DIR = PROJECT_ROOT / "data" / "staging" / "transaction_fraud"

VALIDATION_FILE = DATA_DIR / "transaction_fraud_validation_processed.csv"
MODEL_FILE = MODEL_DIR / "transaction_fraud_best_model.joblib"
THRESHOLD_FILE = OUTPUT_DIR / "transaction_fraud_threshold_metadata.json"

OUTPUT_FILE = OUTPUT_DIR / "transaction_fraud_validation_predictions.csv"


def risk_level(probability):
    if probability >= 0.90:
        return "Critical"
    if probability >= 0.75:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def main():

    print("=" * 70)
    print("TRANSACTION FRAUD INFERENCE PIPELINE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, description in [
        (VALIDATION_FILE, "Validation dataset"),
        (MODEL_FILE, "Best model"),
        (THRESHOLD_FILE, "Threshold metadata"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found:\n{path}"
            )

    print("\nLoading validation dataset...")
    df = pd.read_csv(VALIDATION_FILE)

    target_column = "Class"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column missing: {target_column}"
        )

    X = df.drop(columns=[target_column])

    print(f"Rows     : {len(X):,}")
    print(f"Features : {len(X.columns)}")

    print("\nLoading model...")
    model = joblib.load(MODEL_FILE)

    print(f"Model: {MODEL_FILE}")

    print("\nLoading selected threshold...")

    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    threshold = float(metadata["selected_threshold"])

    print(f"Operating threshold: {threshold:.2f}")

    print("\nGenerating fraud probabilities...")

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = pd.DataFrame({
        "fraud_probability": probabilities,
        "fraud_prediction": predictions,
        "risk_level": [
            risk_level(value)
            for value in probabilities
        ],
        "actual_class": df[target_column].values,
        "model_name": "RandomForest",
        "operating_threshold": threshold,
    })

    print("\nPrediction summary")
    print("-" * 70)

    print(f"Total transactions : {len(result):,}")
    print(
        f"Predicted fraud    : "
        f"{int(result['fraud_prediction'].sum()):,}"
    )
    print(
        f"Predicted legitimate: "
        f"{int((result['fraud_prediction'] == 0).sum()):,}"
    )

    print("\nRisk distribution")
    print(
        result["risk_level"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nInference output saved:")
    print(OUTPUT_FILE)

    print("\nInference pipeline completed successfully.")


if __name__ == "__main__":
    main()
