from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# TRANSACTION FRAUD MODEL TRAINING PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "transaction_fraud"

TRAIN_FILE = DATA_DIR / "transaction_fraud_train_processed.csv"
VALIDATION_FILE = DATA_DIR / "transaction_fraud_validation_processed.csv"
PREPROCESSOR_FILE = DATA_DIR / "transaction_fraud_preprocessor.joblib"

MODEL_DIR = PROJECT_ROOT / "models" / "transaction_fraud"
STAGING_DIR = PROJECT_ROOT / "data" / "staging" / "transaction_fraud"

BEST_MODEL_FILE = MODEL_DIR / "transaction_fraud_best_model.joblib"
MODEL_COMPARISON_FILE = (
    STAGING_DIR / "transaction_fraud_model_comparison.csv"
)
MODEL_EVALUATION_FILE = (
    STAGING_DIR / "transaction_fraud_model_evaluation.json"
)

TARGET_COLUMN = "Class"
RANDOM_STATE = 42


def evaluate_model(model, X_validation, y_validation):

    probabilities = model.predict_proba(X_validation)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
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


def print_metrics(model_name, metrics):

    print(
        f"\nValidation Metrics - {model_name}"
    )

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC   : {metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : {metrics['pr_auc']:.4f}"
    )

    print("\nConfusion Matrix")

    print(
        f"TN: {metrics['tn']}"
    )

    print(
        f"FP: {metrics['fp']}"
    )

    print(
        f"FN: {metrics['fn']}"
    )

    print(
        f"TP: {metrics['tp']}"
    )


def main():

    print("=" * 70)
    print("TRANSACTION FRAUD MODEL TRAINING PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # ========================================================
    # 1. Check artifacts
    # ========================================================

    print("\nChecking required artifacts...")

    required_files = [
        TRAIN_FILE,
        VALIDATION_FILE,
        PREPROCESSOR_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required artifact not found:\n{file_path}"
            )

    print("All required artifacts found.")

    # ========================================================
    # 2. Create directories
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    STAGING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 3. Load datasets
    # ========================================================

    print("\nLoading datasets...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Training rows     : {len(train_df):,}"
    )

    print(
        f"Validation rows   : {len(validation_df):,}"
    )

    print(
        f"Training columns  : {len(train_df.columns)}"
    )

    print(
        f"Validation columns: {len(validation_df.columns)}"
    )

    # ========================================================
    # 4. Target validation
    # ========================================================

    if TARGET_COLUMN not in train_df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in training dataset."
        )

    if TARGET_COLUMN not in validation_df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in validation dataset."
        )

    # ========================================================
    # 5. Split X / y
    # ========================================================

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int)

    X_validation = validation_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_validation = validation_df[
        TARGET_COLUMN
    ].astype(int)

    # ========================================================
    # 6. Feature validation
    # ========================================================

    print("\n## Feature validation")

    print(
        f"Training features  : {X_train.shape[1]}"
    )

    print(
        f"Validation features: {X_validation.shape[1]}"
    )

    if list(X_train.columns) != list(
        X_validation.columns
    ):

        raise ValueError(
            "Training and validation features "
            "are not aligned."
        )

    print(
        f"Feature count: {X_train.shape[1]}"
    )

    print(
        "Training and validation features: ALIGNED"
    )

    # ========================================================
    # 7. Missing / infinite validation
    # ========================================================

    if X_train.isna().any().any():

        raise ValueError(
            "Missing values found in training features."
        )

    if X_validation.isna().any().any():

        raise ValueError(
            "Missing values found in validation features."
        )

    if np.isinf(
        X_train.to_numpy()
    ).any():

        raise ValueError(
            "Infinite values found in training features."
        )

    if np.isinf(
        X_validation.to_numpy()
    ).any():

        raise ValueError(
            "Infinite values found in validation features."
        )

    # ========================================================
    # 8. Target distribution
    # ========================================================

    print("\nTarget distribution")

    print("\nTraining:")

    print(
        y_train.value_counts()
        .sort_index()
    )

    print("\nValidation:")

    print(
        y_validation.value_counts()
        .sort_index()
    )

    # ========================================================
    # 9. Load preprocessor
    # ========================================================

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        f"Preprocessor type: "
        f"{type(preprocessor).__name__}"
    )

    # ========================================================
    # 10. Models
    # ========================================================

    models = {

        "logistic_regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
    }

    results = {}
    trained_models = {}

    # ========================================================
    # 11. Train models
    # ========================================================

    for model_name, model in models.items():

        print("\n" + "=" * 70)

        print(
            f"Training model: {model_name}"
        )

        print("=" * 70)

        print("\nFitting model...")

        model.fit(
            X_train,
            y_train
        )

        print(
            "Model training complete."
        )

        print(
            "\nEvaluating on validation dataset..."
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation
        )

        print_metrics(
            model_name,
            metrics
        )

        results[
            model_name
        ] = metrics

        trained_models[
            model_name
        ] = model

        # ====================================================
        # Save individual model
        # ====================================================

        model_file = (
            MODEL_DIR
            / f"transaction_fraud_{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_file
        )

        print(
            f"\nModel saved: {model_file}"
        )

    # ========================================================
    # 12. Model comparison
    # ========================================================

    comparison_rows = []

    for model_name, metrics in results.items():

        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df = comparison_df.sort_values(
        by="pr_auc",
        ascending=False
    ).reset_index(
        drop=True
    )

    comparison_df.to_csv(
        MODEL_COMPARISON_FILE,
        index=False
    )

    # ========================================================
    # 13. Select best model
    # ========================================================

    best_model_name = (
        comparison_df.iloc[0]["model"]
    )

    best_metrics = results[
        best_model_name
    ]

    best_model = trained_models[
        best_model_name
    ]

    print("\n" + "=" * 70)

    print(
        f"Best model : {best_model_name}"
    )

    print(
        f"PR-AUC     : "
        f"{best_metrics['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC    : "
        f"{best_metrics['roc_auc']:.4f}"
    )

    print(
        f"Recall     : "
        f"{best_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score   : "
        f"{best_metrics['f1_score']:.4f}"
    )

    print("=" * 70)

    # ========================================================
    # 14. Save best model
    # ========================================================

    joblib.dump(
        best_model,
        BEST_MODEL_FILE
    )

    print(
        "\nBest model saved:"
    )

    print(
        BEST_MODEL_FILE
    )

    # ========================================================
    # 15. Save evaluation report
    # ========================================================

    evaluation_report = {

        "pipeline":
            "transaction_fraud_model_training",

        "target_column":
            TARGET_COLUMN,

        "random_state":
            RANDOM_STATE,

        "training_rows":
            int(len(X_train)),

        "validation_rows":
            int(len(X_validation)),

        "feature_count":
            int(X_train.shape[1]),

        "models":
            results,

        "best_model":
            best_model_name,

        "best_model_selection_metric":
            "pr_auc",

        "best_model_metrics":
            best_metrics,
    }

    with open(
        MODEL_EVALUATION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation_report,
            f,
            indent=2
        )

    # ========================================================
    # 16. Final output
    # ========================================================

    print(
        "\nComparison report:"
    )

    print(
        MODEL_COMPARISON_FILE
    )

    print(
        "\nEvaluation report:"
    )

    print(
        MODEL_EVALUATION_FILE
    )

    print(
        "\nModel directory:"
    )

    for model_file in sorted(
        MODEL_DIR.glob("*.joblib")
    ):

        print(
            f" - {model_file.name}"
        )

    print("\n" + "=" * 70)

    print(
        "Transaction fraud model training "
        "pipeline completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()