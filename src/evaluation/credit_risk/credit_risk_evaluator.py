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


# =========================================================
# CONFIGURATION
# =========================================================

TARGET_COLUMN = "SeriousDlqin2yrs"

DEFAULT_THRESHOLD = 0.50

BEST_F1_THRESHOLD = 0.24

BUSINESS_THRESHOLD = 0.18


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_metrics(y_true, probabilities, threshold):
    """
    Calculate classification metrics for a given threshold.
    """

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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(title, metrics):
    """
    Print formatted classification metrics.
    """

    print(f"\n## {title}")
    print()

    print(
        f"Threshold : {metrics['threshold']:.2f}"
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
        f"TN        : {metrics['tn']}"
    )

    print(
        f"FP        : {metrics['fp']}"
    )

    print(
        f"FN        : {metrics['fn']}"
    )

    print(
        f"TP        : {metrics['tp']}"
    )


# =========================================================
# MAIN EVALUATION PIPELINE
# =========================================================

def main():

    print("=" * 70)
    print("CREDIT RISK MODEL EVALUATION PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Validate input files
    # -----------------------------------------------------

    print("\nChecking required artifacts...")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best model not found:\n{MODEL_FILE}"
        )

    # -----------------------------------------------------
    # Load validation dataset
    # -----------------------------------------------------

    print("\nLoading validation dataset...")

    df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Validation rows   : {len(df):,}"
    )

    print(
        f"Validation columns: {len(df.columns)}"
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET_COLUMN}"
        )

    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    print(
        f"Validation features: {X.shape[1]}"
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("\nLoading best model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type        : {type(model).__name__}"
    )

    # -----------------------------------------------------
    # Generate probabilities
    # -----------------------------------------------------

    print(
        "\nGenerating prediction probabilities..."
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise AttributeError(
            "Loaded model does not support predict_proba()."
        )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    # -----------------------------------------------------
    # ROC-AUC
    # -----------------------------------------------------

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    # -----------------------------------------------------
    # PR-AUC
    # -----------------------------------------------------

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    # -----------------------------------------------------
    # Default threshold
    # -----------------------------------------------------

    default_metrics = calculate_metrics(
        y,
        probabilities,
        DEFAULT_THRESHOLD,
    )

    print_metrics(
        "DEFAULT THRESHOLD = 0.50",
        default_metrics,
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    # -----------------------------------------------------
    # Best F1 threshold
    # -----------------------------------------------------

    best_f1_metrics = calculate_metrics(
        y,
        probabilities,
        BEST_F1_THRESHOLD,
    )

    print_metrics(
        "BEST F1 THRESHOLD = 0.24",
        best_f1_metrics,
    )

    # -----------------------------------------------------
    # Business threshold
    # -----------------------------------------------------

    business_metrics = calculate_metrics(
        y,
        probabilities,
        BUSINESS_THRESHOLD,
    )

    print_metrics(
        "BUSINESS THRESHOLD = 0.18",
        business_metrics,
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    business_predictions = (
        probabilities >= BUSINESS_THRESHOLD
    ).astype(int)

    report = classification_report(
        y,
        business_predictions,
        output_dict=True,
        zero_division=0,
    )

    # -----------------------------------------------------
    # Risk segmentation
    # -----------------------------------------------------

    risk_segments = pd.cut(
        probabilities,
        bins=[
            -np.inf,
            0.18,
            0.24,
            0.50,
            np.inf,
        ],
        labels=[
            "Low Risk",
            "Moderate Risk",
            "High Risk",
            "Very High Risk",
        ],
    )

    risk_distribution = (
        risk_segments
        .value_counts()
        .sort_index()
    )

    print("\n## Risk Segment Distribution")
    print()

    for segment, count in risk_distribution.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"{str(segment):15} : "
            f"{count:>7,} "
            f"({percentage:.2f}%)"
        )

    # -----------------------------------------------------
    # Compare thresholds
    # -----------------------------------------------------

    threshold_comparison = pd.DataFrame(
        [
            default_metrics,
            best_f1_metrics,
            business_metrics,
        ]
    )

    threshold_comparison.insert(
        0,
        "strategy",
        [
            "default_0.50",
            "best_f1_0.24",
            "business_0.18",
        ],
    )

    threshold_comparison[
        "roc_auc"
    ] = roc_auc

    threshold_comparison[
        "pr_auc"
    ] = pr_auc

    # -----------------------------------------------------
    # Save threshold comparison
    # -----------------------------------------------------

    threshold_output = (
        OUTPUT_DIR
        / "credit_risk_evaluation_thresholds.csv"
    )

    threshold_comparison.to_csv(
        threshold_output,
        index=False,
    )

    # -----------------------------------------------------
    # Evaluation metadata
    # -----------------------------------------------------

    evaluation = {
        "dataset": str(
            VALIDATION_FILE
        ),
        "model": str(
            MODEL_FILE
        ),
        "model_type": type(model).__name__,
        "validation_rows": int(len(df)),
        "validation_features": int(X.shape[1]),
        "target_column": TARGET_COLUMN,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "default_threshold": default_metrics,
        "best_f1_threshold": best_f1_metrics,
        "business_threshold": business_metrics,
        "risk_segment_distribution": {
            str(key): int(value)
            for key, value
            in risk_distribution.items()
        },
        "classification_report_business_threshold": report,
    }

    # -----------------------------------------------------
    # Save evaluation JSON
    # -----------------------------------------------------

    evaluation_output = (
        OUTPUT_DIR
        / "credit_risk_final_evaluation.json"
    )

    with evaluation_output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Save prediction-level evaluation dataset
    # -----------------------------------------------------

    predictions_output = (
        OUTPUT_DIR
        / "credit_risk_validation_predictions.csv"
    )
    prediction_df = pd.DataFrame(
    {
        "actual_target": y.reset_index(
            drop=True
        ),
        "default_probability": probabilities,
        "prediction_at_0_50": (
            probabilities >= 0.50
        ).astype(int),
        "prediction_at_0_24": (
            probabilities >= 0.24
        ).astype(int),
        "prediction_at_0_18": (
            probabilities >= 0.18
        ).astype(int),
        "risk_segment": pd.Series(
            risk_segments.astype(str),
            index=range(len(risk_segments)),
        ),
    }
)
   

    prediction_df.to_csv(
        predictions_output,
        index=False,
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nROC-AUC              : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC               : {pr_auc:.4f}"
    )

    print(
        f"Default threshold    : {DEFAULT_THRESHOLD:.2f}"
    )

    print(
        f"Best F1 threshold    : {BEST_F1_THRESHOLD:.2f}"
    )

    print(
        f"Business threshold   : {BUSINESS_THRESHOLD:.2f}"
    )

    print("\nArtifacts:")
    print(
        f"Threshold comparison : {threshold_output}"
    )

    print(
        f"Evaluation report    : {evaluation_output}"
    )

    print(
        f"Predictions          : {predictions_output}"
    )

    print(
        "\nEvaluation pipeline completed successfully."
    )


if __name__ == "__main__":
    main()