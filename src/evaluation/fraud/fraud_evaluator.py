from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
# CREDIT / FRAUD MODEL EVALUATION PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud"
    / "fraud_validation_processed.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "fraud"
    / "fraud_best_model.joblib"
)

THRESHOLD_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
    / "fraud_threshold_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
)

THRESHOLD_OUTPUT = OUTPUT_DIR / "fraud_evaluation_thresholds.csv"
EVALUATION_OUTPUT = OUTPUT_DIR / "fraud_final_evaluation.json"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "fraud_validation_predictions.csv"


# ============================================================
# HELPERS
# ============================================================

def ensure_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_true, predictions)
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


def load_threshold_metadata() -> dict:
    if not THRESHOLD_METADATA_FILE.exists():
        return {}

    try:
        with open(
            THRESHOLD_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except Exception:
        return {}


def extract_threshold(
    metadata: dict,
    possible_keys: list[str],
    default: float,
) -> float:

    def recursive_search(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():

                normalized_key = (
                    str(key)
                    .strip()
                    .lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                if normalized_key in possible_keys:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass

                result = recursive_search(value)

                if result is not None:
                    return result

        elif isinstance(obj, list):
            for item in obj:
                result = recursive_search(item)

                if result is not None:
                    return result

        return None

    result = recursive_search(metadata)

    if result is None:
        return default

    return float(result)


def classify_risk(probability: float) -> str:

    if probability < 0.05:
        return "Low Risk"

    if probability < 0.15:
        return "Moderate Risk"

    if probability < 0.30:
        return "High Risk"

    return "Very High Risk"


def business_decision(
    probability: float,
    threshold: float,
) -> str:

    if probability >= threshold:
        return "REVIEW_REQUIRED"

    return "LOW_RISK"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FRAUD MODEL EVALUATION PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Artifact validation
    # --------------------------------------------------------

    print("\nChecking required artifacts...")

    ensure_file_exists(
        VALIDATION_FILE,
        "Validation dataset",
    )

    ensure_file_exists(
        MODEL_FILE,
        "Best fraud model",
    )

    print("All required artifacts found.")

    print("\nValidation dataset:")
    print(VALIDATION_FILE)

    print("\nBest model:")
    print(MODEL_FILE)

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Validation rows    : {len(validation_df):,}"
    )

    print(
        f"Validation columns : {len(validation_df.columns)}"
    )

    target_column = "SeriousDlqin2yrs"

    if target_column not in validation_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    feature_columns = [
        column
        for column in validation_df.columns
        if column != target_column
    ]

    print(
        f"Validation features: {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    y_true = validation_df[target_column].astype(int)

    X_validation = validation_df[
        feature_columns
    ]

    print("\n## Target distribution")
    print()
    print(
        y_true.value_counts().sort_index()
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading best model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type        : {type(model).__name__}"
    )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print("\nGenerating prediction probabilities...")

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Loaded model does not support predict_proba()."
        )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    # --------------------------------------------------------
    # Threshold metadata
    # --------------------------------------------------------

    metadata = load_threshold_metadata()

    best_f1_threshold = extract_threshold(
        metadata,
        [
            "best_f1_threshold",
            "best_f1",
            "optimal_threshold",
            "f1_threshold",
        ],
        0.24,
    )

    business_threshold = extract_threshold(
        metadata,
        [
            "business_threshold",
            "business_oriented_threshold",
            "recall_threshold",
        ],
        0.20,
    )

    default_threshold = 0.50

    # --------------------------------------------------------
    # Default threshold
    # --------------------------------------------------------

    print("\n## DEFAULT THRESHOLD = 0.50")

    default_metrics = calculate_metrics(
        y_true,
        probabilities,
        default_threshold,
    )

    print(
        f"Threshold : {default_metrics['threshold']:.2f}"
    )
    print(
        f"Accuracy  : {default_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision : {default_metrics['precision']:.4f}"
    )
    print(
        f"Recall    : {default_metrics['recall']:.4f}"
    )
    print(
        f"F1 Score  : {default_metrics['f1_score']:.4f}"
    )
    print(
        f"TN        : {default_metrics['tn']}"
    )
    print(
        f"FP        : {default_metrics['fp']}"
    )
    print(
        f"FN        : {default_metrics['fn']}"
    )
    print(
        f"TP        : {default_metrics['tp']}"
    )

    # --------------------------------------------------------
    # Best F1 threshold
    # --------------------------------------------------------

    print(
        f"\n## BEST F1 THRESHOLD = {best_f1_threshold:.2f}"
    )

    best_f1_metrics = calculate_metrics(
        y_true,
        probabilities,
        best_f1_threshold,
    )

    print(
        f"Threshold : {best_f1_metrics['threshold']:.2f}"
    )
    print(
        f"Accuracy  : {best_f1_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision : {best_f1_metrics['precision']:.4f}"
    )
    print(
        f"Recall    : {best_f1_metrics['recall']:.4f}"
    )
    print(
        f"F1 Score  : {best_f1_metrics['f1_score']:.4f}"
    )
    print(
        f"TN        : {best_f1_metrics['tn']}"
    )
    print(
        f"FP        : {best_f1_metrics['fp']}"
    )
    print(
        f"FN        : {best_f1_metrics['fn']}"
    )
    print(
        f"TP        : {best_f1_metrics['tp']}"
    )

    # --------------------------------------------------------
    # Business threshold
    # --------------------------------------------------------

    print(
        f"\n## BUSINESS THRESHOLD = {business_threshold:.2f}"
    )

    business_metrics = calculate_metrics(
        y_true,
        probabilities,
        business_threshold,
    )

    print(
        f"Threshold : {business_metrics['threshold']:.2f}"
    )
    print(
        f"Accuracy  : {business_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision : {business_metrics['precision']:.4f}"
    )
    print(
        f"Recall    : {business_metrics['recall']:.4f}"
    )
    print(
        f"F1 Score  : {business_metrics['f1_score']:.4f}"
    )
    print(
        f"TN        : {business_metrics['tn']}"
    )
    print(
        f"FP        : {business_metrics['fp']}"
    )
    print(
        f"FN        : {business_metrics['fn']}"
    )
    print(
        f"TP        : {business_metrics['tp']}"
    )

    # --------------------------------------------------------
    # ROC-AUC / PR-AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    print("\n## Model-Level Metrics")

    print(
        f"ROC-AUC              : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC               : {pr_auc:.4f}"
    )

    print(
        f"Default threshold    : {default_threshold:.2f}"
    )

    print(
        f"Best F1 threshold    : {best_f1_threshold:.2f}"
    )

    print(
        f"Business threshold   : {business_threshold:.2f}"
    )

    # --------------------------------------------------------
    # Risk segmentation
    # --------------------------------------------------------

    risk_segments = pd.Series(
        [
            classify_risk(probability)
            for probability in probabilities
        ],
        name="risk_segment",
    )

    print("\n## Risk Segment Distribution")

    risk_counts = (
        risk_segments
        .value_counts()
        .reindex(
            [
                "Low Risk",
                "Moderate Risk",
                "High Risk",
                "Very High Risk",
            ],
            fill_value=0,
        )
    )

    for segment, count in risk_counts.items():

        percentage = (
            count / len(risk_segments)
        ) * 100

        print(
            f"{segment:<16}: {count:6,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Prediction artifact
    # --------------------------------------------------------

    default_predictions = (
        probabilities >= default_threshold
    ).astype(int)

    best_f1_predictions = (
        probabilities >= best_f1_threshold
    ).astype(int)

    business_predictions = (
        probabilities >= business_threshold
    ).astype(int)

    predictions_df = pd.DataFrame(
        {
            "actual_target": y_true.to_numpy(),
            "default_probability": probabilities,
            "prediction_at_0_50": default_predictions,
            "prediction_at_best_f1": best_f1_predictions,
            "prediction_at_business_threshold": business_predictions,
            "risk_segment": risk_segments,
            "credit_decision": [
                business_decision(
                    probability,
                    business_threshold,
                )
                for probability in probabilities
            ],
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Threshold comparison
    # --------------------------------------------------------

    threshold_comparison = pd.DataFrame(
        [
            {
                "model": "gradient_boosting",
                "threshold_type": "default",
                **default_metrics,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
            {
                "model": "gradient_boosting",
                "threshold_type": "best_f1",
                **best_f1_metrics,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
            {
                "model": "gradient_boosting",
                "threshold_type": "business",
                **business_metrics,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
        ]
    )

    threshold_comparison.to_csv(
        THRESHOLD_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Final evaluation report
    # --------------------------------------------------------

    evaluation_report = {
        "project": "Enterprise Banking Risk Intelligence Platform",
        "pipeline": "Fraud Model Evaluation",
        "model": {
            "type": type(model).__name__,
            "file": str(MODEL_FILE),
        },
        "dataset": {
            "validation_file": str(VALIDATION_FILE),
            "rows": int(len(validation_df)),
            "columns": int(len(validation_df.columns)),
            "features": int(len(feature_columns)),
        },
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
        "thresholds": {
            "default": float(default_threshold),
            "best_f1": float(best_f1_threshold),
            "business": float(business_threshold),
        },
        "default_threshold_metrics": default_metrics,
        "best_f1_threshold_metrics": best_f1_metrics,
        "business_threshold_metrics": business_metrics,
        "risk_segments": {
            segment: {
                "count": int(count),
                "percentage": float(
                    count / len(risk_segments) * 100
                ),
            }
            for segment, count in risk_counts.items()
        },
        "artifacts": {
            "threshold_comparison": str(
                THRESHOLD_OUTPUT
            ),
            "predictions": str(
                PREDICTIONS_OUTPUT
            ),
            "evaluation_report": str(
                EVALUATION_OUTPUT
            ),
        },
    }

    with open(
        EVALUATION_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation_report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ARTIFACTS")
    print("=" * 70)

    print(
        f"\nThreshold comparison : {THRESHOLD_OUTPUT}"
    )

    print(
        f"Evaluation report    : {EVALUATION_OUTPUT}"
    )

    print(
        f"Predictions          : {PREDICTIONS_OUTPUT}"
    )

    print(
        "\nFraud evaluation pipeline completed successfully."
    )


if __name__ == "__main__":
    main()