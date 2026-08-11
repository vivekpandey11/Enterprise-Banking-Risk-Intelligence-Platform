from pathlib import Path
import json
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from joblib import dump


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "aml" / "aml_train_processed.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "aml" / "aml_validation_processed.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "aml"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("\n# AML MODEL TRAINING PIPELINE\n")

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALIDATION_FILE)

print(f"Training rows   : {len(train_df):,}")
print(f"Validation rows : {len(valid_df):,}")

TARGET = "is_laundering"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_valid = valid_df.drop(columns=[TARGET])
y_valid = valid_df[TARGET]

print("\nTarget distribution (training)")
print(y_train.value_counts())

models = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    try:
        probabilities = model.predict_proba(X_valid)[:, 1]
    except Exception:
        probabilities = None

    accuracy = accuracy_score(y_valid, predictions)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0
    )

    if probabilities is not None and len(set(y_valid)) > 1:
        roc_auc = roc_auc_score(y_valid, probabilities)
    else:
        roc_auc = None

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    model_path = MODEL_DIR / f"aml_{model_name}.joblib"
    dump(model, model_path)

    print(f"Saved: {model_path}")

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    })

results_df = pd.DataFrame(results)

comparison_file = PROJECT_ROOT / "data" / "staging" / "aml" / "aml_model_comparison.csv"
comparison_file.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(comparison_file, index=False)

best_model_name = results_df.sort_values(
    by="f1",
    ascending=False
).iloc[0]["model"]

best_model_file = MODEL_DIR / f"aml_{best_model_name}.joblib"

metadata = {
    "best_model": best_model_name,
    "best_model_path": str(best_model_file)
}

metadata_file = PROJECT_ROOT / "data" / "staging" / "aml" / "aml_model_metadata.json"

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nBest model:", best_model_name)
print("Comparison :", comparison_file)
print("Metadata   :", metadata_file)

print("\nAML model training pipeline completed successfully.")