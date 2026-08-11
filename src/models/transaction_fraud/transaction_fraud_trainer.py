from pathlib import Path
import json
import warnings

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "transaction_fraud"
MODEL_DIR = PROJECT_ROOT / "models" / "transaction_fraud"
OUTPUT_DIR = PROJECT_ROOT / "data" / "staging" / "transaction_fraud"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "transaction_fraud_train_processed.csv"
VALIDATION_FILE = DATA_DIR / "transaction_fraud_validation_processed.csv"

LOGISTIC_MODEL_FILE = MODEL_DIR / "transaction_fraud_logistic_regression.joblib"
RANDOM_FOREST_MODEL_FILE = MODEL_DIR / "transaction_fraud_random_forest.joblib"
BEST_MODEL_FILE = MODEL_DIR / "transaction_fraud_best_model.joblib"

MODEL_COMPARISON_FILE = OUTPUT_DIR / "transaction_fraud_model_comparison.csv"
MODEL_EVALUATION_FILE = OUTPUT_DIR / "transaction_fraud_model_evaluation.json"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def evaluate_model(model, X_valid, y_valid):
    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(y_valid, predictions)
    precision = precision_score(
        y_valid, predictions, zero_division=0
    )
    recall = recall_score(
        y_valid, predictions, zero_division=0
    )
    f1 = f1_score(
        y_valid, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(
        y_valid, probabilities
    )
    pr_auc = average_precision_score(
        y_valid, probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():

    print("=" * 70)
    print("TRANSACTION FRAUD MODEL TRAINING PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nChecking required files...")

    require_file(TRAIN_FILE, "Training dataset")
    require_file(VALIDATION_FILE, "Validation dataset")

    print("Required files: OK")

    print("\nLoading processed datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALIDATION_FILE)

    print(f"Training rows   : {len(train_df):,}")
    print(f"Validation rows : {len(valid_df):,}")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column missing from training data: {TARGET_COLUMN}"
        )

    if TARGET_COLUMN not in valid_df.columns:
        raise ValueError(
            f"Target column missing from validation data: {TARGET_COLUMN}"
        )

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_valid = valid_df.drop(columns=[TARGET_COLUMN])
    y_valid = valid_df[TARGET_COLUMN]

    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError(
            "Training and validation features are not aligned."
        )

    print(f"\nFeature count: {X_train.shape[1]}")
    print("Training/validation features: ALIGNED")

    print("\nTarget distribution")

    print("\nTraining:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nValidation:")
    print(y_valid.value_counts().sort_index().to_string())

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    evaluation_details = {}
    trained_models = {}

    for model_name, model in models.items():

        print("\n" + "=" * 70)
        print(f"TRAINING: {model_name.upper()}")
        print("=" * 70)

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_valid,
            y_valid,
        )

        trained_models[model_name] = model
        evaluation_details[model_name] = metrics

        result = {
            "model": model_name,
            **metrics,
        }

        results.append(result)

        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1 Score  : {metrics['f1_score']:.4f}")
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
        print(
            f"Confusion : "
            f"TN={metrics['tn']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']} "
            f"TP={metrics['tp']}"
        )

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by=["pr_auc", "recall", "f1_score"],
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df[
            [
                "model",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    best_model_name = comparison_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print("\nSelected best model:")
    print(best_model_name)

    joblib.dump(
        trained_models["logistic_regression"],
        LOGISTIC_MODEL_FILE,
    )

    joblib.dump(
        trained_models["random_forest"],
        RANDOM_FOREST_MODEL_FILE,
    )

    joblib.dump(
        best_model,
        BEST_MODEL_FILE,
    )

    comparison_df.to_csv(
        MODEL_COMPARISON_FILE,
        index=False,
    )

    evaluation_metadata = {
        "project": "Enterprise Banking Risk Intelligence Platform",
        "pipeline": "Transaction Fraud Detection",
        "target_column": TARGET_COLUMN,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "feature_count": int(X_train.shape[1]),
        "best_model": best_model_name,
        "selection_metric": "PR-AUC",
        "models": evaluation_details,
        "output_models": {
            "logistic_regression": str(LOGISTIC_MODEL_FILE),
            "random_forest": str(RANDOM_FOREST_MODEL_FILE),
            "best_model": str(BEST_MODEL_FILE),
        },
        "model_comparison": str(MODEL_COMPARISON_FILE),
    }

    with open(
        MODEL_EVALUATION_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_metadata,
            file,
            indent=2,
        )

    print("\nModels saved:")
    print(LOGISTIC_MODEL_FILE)
    print(RANDOM_FOREST_MODEL_FILE)
    print(BEST_MODEL_FILE)

    print("\nReports saved:")
    print(MODEL_COMPARISON_FILE)
    print(MODEL_EVALUATION_FILE)

    print("\n" + "=" * 70)
    print("TRANSACTION FRAUD MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
