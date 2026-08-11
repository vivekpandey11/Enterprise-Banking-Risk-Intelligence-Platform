from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "credit_risk"
    / "credit_risk_validation_processed.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "credit_risk"
    / "credit_risk_best_model.joblib"
)

THRESHOLD_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
    / "credit_risk_threshold_metadata.json"
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

TARGET_COLUMN = "SeriousDlqin2yrs"

DEFAULT_THRESHOLD = 0.50


# =========================================================
# LOAD VALIDATION DATA
# =========================================================

def load_validation_data():

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_FILE}"
        )

    df = pd.read_csv(
        VALIDATION_FILE
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET_COLUMN}"
        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    return X, y


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best model not found:\n{MODEL_FILE}"
        )

    return joblib.load(
        MODEL_FILE
    )


# =========================================================
# LOAD THRESHOLD METADATA
# =========================================================

def load_threshold():

    if not THRESHOLD_METADATA_FILE.exists():
        print(
            "\nThreshold metadata not found."
        )
        print(
            "Using default threshold: 0.50"
        )

        return DEFAULT_THRESHOLD

    with THRESHOLD_METADATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    threshold = metadata.get(
        "business_threshold",
        DEFAULT_THRESHOLD,
    )

    return float(threshold)


# =========================================================
# GET POSITIVE CLASS PROBABILITY
# =========================================================

def get_probabilities(
    model,
    X,
):

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise ValueError(
            "Model does not support predict_proba()."
        )

    probabilities = model.predict_proba(
        X
    )

    if probabilities.shape[1] != 2:
        raise ValueError(
            "Expected binary classification model."
        )

    return probabilities[:, 1]


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def extract_feature_importance(
    model,
    feature_names,
):

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return []

    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        print(
            "\nWARNING: Feature importance count "
            "does not match feature count."
        )

        return []

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df[
        "importance_percentage"
    ] = (
        importance_df["importance"]
        / importance_df["importance"].sum()
        * 100
    )

    return importance_df


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CREDIT RISK MODEL EVALUATION PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("\nLoading validation dataset...")

    X_valid, y_valid = load_validation_data()

    print(
        f"Validation rows   : {len(X_valid):,}"
    )

    print(
        f"Validation features: {X_valid.shape[1]}"
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("\nLoading trained model...")

    model = load_model()

    print(
        f"Model type        : {type(model).__name__}"
    )

    print(
        f"Model file        : {MODEL_FILE}"
    )

    # -----------------------------------------------------
    # Generate probabilities
    # -----------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = get_probabilities(
        model,
        X_valid,
    )

    # -----------------------------------------------------
    # Thresholds
    # -----------------------------------------------------

    business_threshold = load_threshold()

    print(
        f"\nDefault threshold  : "
        f"{DEFAULT_THRESHOLD:.2f}"
    )

    print(
        f"Business threshold : "
        f"{business_threshold:.2f}"
    )

    # -----------------------------------------------------
    # Evaluate default threshold
    # -----------------------------------------------------

    default_metrics = calculate_metrics(
        y_valid,
        probabilities,
        DEFAULT_THRESHOLD,
    )

    # -----------------------------------------------------
    # Evaluate business threshold
    # -----------------------------------------------------

    business_metrics = calculate_metrics(
        y_valid,
        probabilities,
        business_threshold,
    )

    # -----------------------------------------------------
    # Print default metrics
    # -----------------------------------------------------

    print("\nDEFAULT THRESHOLD EVALUATION")
    print("-" * 70)

    print(
        f"Accuracy  : "
        f"{default_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{default_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{default_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{default_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{default_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{default_metrics['pr_auc']:.4f}"
    )

    print("\nConfusion Matrix")
    print(
        f"TN: {default_metrics['true_negative']}"
    )
    print(
        f"FP: {default_metrics['false_positive']}"
    )
    print(
        f"FN: {default_metrics['false_negative']}"
    )
    print(
        f"TP: {default_metrics['true_positive']}"
    )

    # -----------------------------------------------------
    # Print business threshold metrics
    # -----------------------------------------------------

    print("\nBUSINESS THRESHOLD EVALUATION")
    print("-" * 70)

    print(
        f"Threshold : "
        f"{business_threshold:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{business_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{business_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{business_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{business_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{business_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{business_metrics['pr_auc']:.4f}"
    )

    print("\nConfusion Matrix")
    print(
        f"TN: {business_metrics['true_negative']}"
    )
    print(
        f"FP: {business_metrics['false_positive']}"
    )
    print(
        f"FN: {business_metrics['false_negative']}"
    )
    print(
        f"TP: {business_metrics['true_positive']}"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    business_predictions = (
        probabilities >= business_threshold
    ).astype(int)

    classification_report_text = (
        classification_report(
            y_valid,
            business_predictions,
            target_names=[
                "Non-Default",
                "Default",
            ],
            zero_division=0,
        )
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)
    print(
        classification_report_text
    )

    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

    feature_names = list(
        X_valid.columns
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    importance_df = (
        extract_feature_importance(
            model,
            feature_names,
        )
    )

    if isinstance(
        importance_df,
        pd.DataFrame,
    ):

        importance_file = (
            OUTPUT_DIR
            / "credit_risk_feature_importance.csv"
        )

        importance_df.to_csv(
            importance_file,
            index=False,
        )

        print(
            "\nTOP 15 FEATURES"
        )

        print("-" * 70)

        print(
            importance_df
            .head(15)
            .to_string(index=False)
        )

    else:

        importance_file = None

        print(
            "\nFeature importance "
            "not available."
        )

    # -----------------------------------------------------
    # Evaluation comparison
    # -----------------------------------------------------

    comparison_df = pd.DataFrame(
        [
            {
                "strategy": "default_0.50",
                **default_metrics,
            },
            {
                "strategy": "business_threshold",
                **business_metrics,
            },
        ]
    )

    comparison_file = (
        OUTPUT_DIR
        / "credit_risk_evaluation_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    # -----------------------------------------------------
    # Evaluation report
    # -----------------------------------------------------

    evaluation_report = {
        "dataset": str(
            VALIDATION_FILE
        ),
        "model": str(
            MODEL_FILE
        ),
        "model_type": type(
            model
        ).__name__,
        "validation_rows": int(
            len(X_valid)
        ),
        "feature_count": int(
            X_valid.shape[1]
        ),
        "target_column": TARGET_COLUMN,
        "default_threshold": DEFAULT_THRESHOLD,
        "business_threshold": business_threshold,
        "default_threshold_metrics": (
            default_metrics
        ),
        "business_threshold_metrics": (
            business_metrics
        ),
        "classification_report": (
            classification_report(
                y_valid,
                business_predictions,
                output_dict=True,
                zero_division=0,
            )
        ),
        "feature_importance_file": (
            str(importance_file)
            if importance_file
            else None
        ),
        "evaluation_comparison_file": str(
            comparison_file
        ),
    }

    evaluation_file = (
        OUTPUT_DIR
        / "credit_risk_evaluation_report.json"
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
    # Final summary
    # -----------------------------------------------------

    print("\nEVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Model                : "
        f"{type(model).__name__}"
    )

    print(
        f"ROC-AUC              : "
        f"{business_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC               : "
        f"{business_metrics['pr_auc']:.4f}"
    )

    print(
        f"Business threshold   : "
        f"{business_threshold:.2f}"
    )

    print(
        f"Business precision   : "
        f"{business_metrics['precision']:.4f}"
    )

    print(
        f"Business recall      : "
        f"{business_metrics['recall']:.4f}"
    )

    print(
        f"Business F1          : "
        f"{business_metrics['f1_score']:.4f}"
    )

    print("\nArtifacts:")
    print(
        f"Evaluation report    : "
        f"{evaluation_file}"
    )

    print(
        f"Comparison report    : "
        f"{comparison_file}"
    )

    if importance_file:
        print(
            f"Feature importance   : "
            f"{importance_file}"
        )

    print(
        "\nModel evaluation pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()