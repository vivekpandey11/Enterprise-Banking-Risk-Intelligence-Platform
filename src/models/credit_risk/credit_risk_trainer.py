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
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
# FILES
# =========================================================

VALIDATION_FILE = (
    DATA_DIR
    / "credit_risk_validation_processed.csv"
)

BEST_MODEL_FILE = (
    MODEL_DIR
    / "credit_risk_best_model.joblib"
)

MODEL_COMPARISON_FILE = (
    OUTPUT_DIR
    / "credit_risk_model_comparison.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

TARGET_COLUMN = "SeriousDlqin2yrs"

THRESHOLDS = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05,
]

# Minimum recall desired for risk detection.
#
# This is not a universal banking rule.
# It is a project-level operating target used to
# demonstrate threshold selection.
MIN_RECALL = 0.70


# =========================================================
# LOAD VALIDATION DATA
# =========================================================

def load_validation_data():

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n"
            f"{VALIDATION_FILE}"
        )

    df = pd.read_csv(
        VALIDATION_FILE
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column not found: "
            f"{TARGET_COLUMN}"
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

    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best model not found:\n"
            f"{BEST_MODEL_FILE}"
        )

    model = joblib.load(
        BEST_MODEL_FILE
    )

    return model


# =========================================================
# EVALUATE THRESHOLD
# =========================================================

def evaluate_threshold(
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

    total_positive_predictions = (
        int(predictions.sum())
    )

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
        "predicted_high_risk": (
            total_positive_predictions
        ),
    }


# =========================================================
# SELECT OPERATING THRESHOLD
# =========================================================

def select_threshold(results):

    eligible = [
        result
        for result in results
        if result["recall"] >= MIN_RECALL
    ]

    if eligible:

        # Among thresholds satisfying the recall
        # requirement, select the one with the
        # highest F1 score.

        selected = max(
            eligible,
            key=lambda result: (
                result["f1_score"],
                result["precision"],
            ),
        )

        selection_reason = (
            f"Selected highest-F1 threshold "
            f"among thresholds with recall >= "
            f"{MIN_RECALL:.2f}."
        )

    else:

        # If no threshold satisfies the recall
        # requirement, choose the threshold with
        # highest recall.

        selected = max(
            results,
            key=lambda result: (
                result["recall"],
                result["f1_score"],
            ),
        )

        selection_reason = (
            "No threshold satisfied the minimum "
            "recall requirement. Selected the "
            "threshold with highest recall."
        )

    return selected, selection_reason


# =========================================================
# RISK CATEGORY
# =========================================================

def risk_category(probability):

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.40:
        return "MEDIUM"

    return "LOW"


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CREDIT RISK MODEL THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load validation dataset
    # -----------------------------------------------------

    print("\nLoading validation dataset...")

    X_valid, y_valid = load_validation_data()

    print(
        f"Validation rows   : {len(X_valid):,}"
    )

    print(
        f"Validation features: {len(X_valid.columns)}"
    )

    # -----------------------------------------------------
    # Load best model
    # -----------------------------------------------------

    print("\nLoading best model...")

    model = load_model()

    print(
        f"Model file:\n{BEST_MODEL_FILE}"
    )

    # -----------------------------------------------------
    # Generate probabilities
    # -----------------------------------------------------

    print("\nGenerating risk probabilities...")

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise AttributeError(
            "Loaded model does not support "
            "predict_proba()."
        )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    print(
        f"Minimum probability : "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Maximum probability : "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Mean probability    : "
        f"{probabilities.mean():.6f}"
    )

    # -----------------------------------------------------
    # Evaluate thresholds
    # -----------------------------------------------------

    print("\nThreshold evaluation")
    print("-" * 70)

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            y_true=y_valid,
            probabilities=probabilities,
            threshold=threshold,
        )

        results.append(
            result
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1_score']:.4f} | "
            f"PR-AUC={result['pr_auc']:.4f}"
        )

    # -----------------------------------------------------
    # Convert results to dataframe
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    threshold_report = (
        OUTPUT_DIR
        / "credit_risk_threshold_analysis.csv"
    )

    results_df.to_csv(
        threshold_report,
        index=False,
    )

    # -----------------------------------------------------
    # Select threshold
    # -----------------------------------------------------

    selected_threshold, selection_reason = (
        select_threshold(results)
    )

    # -----------------------------------------------------
    # Risk category example
    # -----------------------------------------------------

    example_probability = float(
        np.mean(probabilities)
    )

    example_category = risk_category(
        example_probability
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "model_file": str(
            BEST_MODEL_FILE
        ),
        "validation_file": str(
            VALIDATION_FILE
        ),
        "target_column": TARGET_COLUMN,
        "validation_rows": int(
            len(X_valid)
        ),
        "thresholds_evaluated": [
            float(value)
            for value in THRESHOLDS
        ],
        "minimum_recall_target": (
            float(MIN_RECALL)
        ),
        "selected_threshold": float(
            selected_threshold["threshold"]
        ),
        "selection_reason": selection_reason,
        "selected_metrics": selected_threshold,
        "probability_summary": {
            "minimum": float(
                probabilities.min()
            ),
            "maximum": float(
                probabilities.max()
            ),
            "mean": float(
                probabilities.mean()
            ),
            "median": float(
                np.median(probabilities)
            ),
        },
        "risk_bands": {
            "LOW": "< 0.40",
            "MEDIUM": "0.40 - 0.69",
            "HIGH": ">= 0.70",
        },
        "example_probability": (
            example_probability
        ),
        "example_risk_category": (
            example_category
        ),
    }

    metadata_file = (
        OUTPUT_DIR
        / "credit_risk_threshold_metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSelected threshold : "
        f"{selected_threshold['threshold']:.2f}"
    )

    print(
        f"Precision          : "
        f"{selected_threshold['precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{selected_threshold['recall']:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{selected_threshold['f1_score']:.4f}"
    )

    print(
        f"PR-AUC             : "
        f"{selected_threshold['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC            : "
        f"{selected_threshold['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix")
    print("-" * 70)

    print(
        f"TN: "
        f"{selected_threshold['true_negative']:,}"
    )

    print(
        f"FP: "
        f"{selected_threshold['false_positive']:,}"
    )

    print(
        f"FN: "
        f"{selected_threshold['false_negative']:,}"
    )

    print(
        f"TP: "
        f"{selected_threshold['true_positive']:,}"
    )

    print(
        "\nThreshold analysis report:"
    )

    print(
        threshold_report
    )

    print(
        "\nThreshold metadata:"
    )

    print(
        metadata_file
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()