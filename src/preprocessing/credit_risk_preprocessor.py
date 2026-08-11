from pathlib import Path
import json
import joblib

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "credit_risk"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "credit_risk"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# INPUT FILES
# =========================================================

TRAIN_FILE = (
    DATA_DIR
    / "credit_risk_train_processed.csv"
)

VALIDATION_FILE = (
    DATA_DIR
    / "credit_risk_validation_processed.csv"
)

TARGET_COLUMN = "SeriousDlqin2yrs"

RANDOM_STATE = 42


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def build_models():

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
    }

    return models


# =========================================================
# METRIC CALCULATION
# =========================================================

def evaluate_model(
    model,
    X,
    y,
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    matrix = confusion_matrix(
        y,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


# =========================================================
# MAIN TRAINING PIPELINE
# =========================================================

def main():

    print("=" * 70)
    print("CREDIT RISK ML MODEL TRAINING PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Validate input files
    # -----------------------------------------------------

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_FILE}"
        )

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    print("\nLoading datasets...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Training rows   : {len(train_df):,}"
    )

    print(
        f"Validation rows : {len(validation_df):,}"
    )

    print(
        f"Training columns   : {len(train_df.columns)}"
    )

    print(
        f"Validation columns : {len(validation_df.columns)}"
    )

    # -----------------------------------------------------
    # Validate target
    # -----------------------------------------------------

    if TARGET_COLUMN not in train_df.columns:

        raise ValueError(
            f"Target column missing from training data: "
            f"{TARGET_COLUMN}"
        )

    if TARGET_COLUMN not in validation_df.columns:

        raise ValueError(
            f"Target column missing from validation data: "
            f"{TARGET_COLUMN}"
        )

    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    # -----------------------------------------------------
    # Validate feature alignment
    # -----------------------------------------------------

    if list(X_train.columns) != list(
        X_validation.columns
    ):

        raise ValueError(
            "Training and validation feature columns "
            "are not aligned."
        )

    print("\nFeature validation")
    print("-" * 70)

    print(
        f"Feature count : {X_train.shape[1]}"
    )

    print(
        "Training and validation feature columns: ALIGNED"
    )

    # -----------------------------------------------------
    # Target distribution
    # -----------------------------------------------------

    print("\nTarget distribution")
    print("-" * 70)

    print(
        "Training:"
    )

    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nValidation:"
    )

    print(
        y_validation.value_counts()
        .sort_index()
        .to_string()
    )

    # -----------------------------------------------------
    # Build models
    # -----------------------------------------------------

    models = build_models()

    results = {}

    trained_models = {}

    # -----------------------------------------------------
    # Train each model
    # -----------------------------------------------------

    for model_name, model in models.items():

        print("\n")
        print("=" * 70)
        print(
            f"TRAINING MODEL: {model_name}"
        )
        print("=" * 70)

        print(
            "\nFitting model..."
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Model training complete."
        )

        # -------------------------------------------------
        # Training metrics
        # -------------------------------------------------

        print(
            "\nEvaluating on validation dataset..."
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        results[model_name] = metrics

        trained_models[model_name] = model

        print("\nValidation Metrics")
        print("-" * 70)

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
            f"TN: {metrics['true_negatives']}"
        )

        print(
            f"FP: {metrics['false_positives']}"
        )

        print(
            f"FN: {metrics['false_negatives']}"
        )

        print(
            f"TP: {metrics['true_positives']}"
        )

    # -----------------------------------------------------
    # Model comparison
    #
    # PR-AUC is especially important for imbalanced
    # credit-risk classification.
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

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
        ascending=False,
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Select best model
    #
    # PR-AUC is used because the positive class is
    # significantly smaller than the negative class.
    # -----------------------------------------------------

    best_model_name = (
        comparison_df.iloc[0]["model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_metrics = results[
        best_model_name
    ]

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Model    : {best_model_name}"
    )

    print(
        f"PR-AUC   : {best_metrics['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC  : {best_metrics['roc_auc']:.4f}"
    )

    print(
        f"Recall   : {best_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {best_metrics['f1_score']:.4f}"
    )

    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    model_file = (
        MODEL_DIR
        / "credit_risk_best_model.joblib"
    )

    joblib.dump(
        best_model,
        model_file,
    )

    print("\nBest model saved:")
    print(model_file)

    # -----------------------------------------------------
    # Save individual models
    # -----------------------------------------------------

    for model_name, model in trained_models.items():

        model_file_individual = (
            MODEL_DIR
            / f"credit_risk_{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_file_individual,
        )

    print(
        "\nIndividual model artifacts saved."
    )

    # -----------------------------------------------------
    # Save model comparison
    # -----------------------------------------------------

    comparison_file = (
        REPORT_DIR
        / "credit_risk_model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    # -----------------------------------------------------
    # Save evaluation report
    # -----------------------------------------------------

    evaluation_report = {
        "dataset": {
            "training_file": str(TRAIN_FILE),
            "validation_file": str(VALIDATION_FILE),
            "training_rows": int(len(train_df)),
            "validation_rows": int(
                len(validation_df)
            ),
            "feature_count": int(
                X_train.shape[1]
            ),
            "target_column": TARGET_COLUMN,
        },
        "training_configuration": {
            "random_state": RANDOM_STATE,
            "models_trained": list(
                models.keys()
            ),
            "model_selection_metric": "pr_auc",
            "reason": (
                "PR-AUC is used as the primary "
                "selection metric because the "
                "credit-risk target is imbalanced."
            ),
        },
        "models": results,
        "best_model": {
            "name": best_model_name,
            "metrics": best_metrics,
            "artifact": str(
                MODEL_DIR
                / "credit_risk_best_model.joblib"
            ),
        },
    }

    evaluation_file = (
        REPORT_DIR
        / "credit_risk_model_evaluation.json"
    )

    with evaluation_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation_report,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest model        : {best_model_name}"
    )

    print(
        f"Best PR-AUC       : "
        f"{best_metrics['pr_auc']:.4f}"
    )

    print(
        f"Best ROC-AUC      : "
        f"{best_metrics['roc_auc']:.4f}"
    )

    print(
        f"\nBest model file   : {model_file}"
    )

    print(
        f"Comparison report : {comparison_file}"
    )

    print(
        f"Evaluation report : {evaluation_file}"
    )

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()