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
# PROJECT PATHS
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
)

THRESHOLD_RESULTS_FILE = (
    OUTPUT_DIR / "fraud_threshold_results.csv"
)

THRESHOLD_COMPARISON_FILE = (
    OUTPUT_DIR / "fraud_threshold_comparison.csv"
)

THRESHOLD_METADATA_FILE = (
    OUTPUT_DIR / "fraud_threshold_metadata.json"
)


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

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("FRAUD THRESHOLD OPTIMIZATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Check artifacts
    # --------------------------------------------------------

    print("\nChecking required artifacts...")

    ensure_file_exists(
        VALIDATION_FILE,
        "Fraud validation dataset",
    )

    ensure_file_exists(
        MODEL_FILE,
        "Fraud best model",
    )

    print("All required artifacts found.")

    print(f"\nValidation dataset:")
    print(VALIDATION_FILE)

    print(f"\nModel:")
    print(MODEL_FILE)

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    df = pd.read_csv(VALIDATION_FILE)

    print(f"Validation rows    : {len(df):,}")
    print(f"Validation columns : {len(df.columns)}")

    target_column = "SeriousDlqin2yrs"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    y = df[target_column].astype(int)

    X = df.drop(
        columns=[target_column]
    )

    print(f"Validation features: {X.shape[1]}")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\n## Target distribution")

    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading best model...")

    model = joblib.load(MODEL_FILE)

    print(
        f"Model type        : {type(model).__name__}"
    )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating prediction probabilities..."
    )

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Loaded model does not support predict_proba()."
        )

    probabilities = model.predict_proba(X)[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(probabilities) != len(y):
        raise ValueError(
            "Prediction count does not match validation rows."
        )

    # --------------------------------------------------------
    # ROC-AUC / PR-AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    # --------------------------------------------------------
    # Default threshold
    # --------------------------------------------------------

    DEFAULT_THRESHOLD = 0.50

    default_metrics = calculate_metrics(
        y,
        probabilities,
        DEFAULT_THRESHOLD,
    )

    print("\n## DEFAULT THRESHOLD = 0.50")

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
    # Search thresholds
    # --------------------------------------------------------

    print(
        "\n## Searching optimal classification threshold..."
    )

    thresholds = np.round(
        np.arange(
            0.01,
            1.00,
            0.01,
        ),
        2,
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y,
            probabilities,
            float(threshold),
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Best F1 threshold
    # --------------------------------------------------------

    best_f1_row = (
        results_df
        .sort_values(
            by=[
                "f1_score",
                "recall",
                "precision",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    best_f1_threshold = float(
        best_f1_row["threshold"]
    )

    # --------------------------------------------------------
    # Business threshold
    #
    # Fraud detection generally prioritizes recall.
    # Select the highest-precision threshold that still
    # achieves recall >= 0.50.
    # --------------------------------------------------------

    business_candidates = results_df[
        results_df["recall"] >= 0.50
    ].copy()

    if business_candidates.empty:

        business_row = (
            results_df
            .sort_values(
                by=[
                    "recall",
                    "precision",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        business_threshold = float(
            business_row["threshold"]
        )

    else:

        business_row = (
            business_candidates
            .sort_values(
                by=[
                    "precision",
                    "f1_score",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        business_threshold = float(
            business_row["threshold"]
        )

    # --------------------------------------------------------
    # Print best F1
    # --------------------------------------------------------

    print("\n## BEST F1 THRESHOLD")

    print(
        f"Threshold : {best_f1_threshold:.2f}"
    )
    print(
        f"Accuracy  : {best_f1_row['accuracy']:.4f}"
    )
    print(
        f"Precision : {best_f1_row['precision']:.4f}"
    )
    print(
        f"Recall    : {best_f1_row['recall']:.4f}"
    )
    print(
        f"F1 Score  : {best_f1_row['f1_score']:.4f}"
    )
    print(
        f"TN        : {int(best_f1_row['tn'])}"
    )
    print(
        f"FP        : {int(best_f1_row['fp'])}"
    )
    print(
        f"FN        : {int(best_f1_row['fn'])}"
    )
    print(
        f"TP        : {int(best_f1_row['tp'])}"
    )

    # --------------------------------------------------------
    # Print business threshold
    # --------------------------------------------------------

    print("\n## BUSINESS-ORIENTED THRESHOLD")

    print("Recall target : >= 0.50")

    print(
        f"Threshold     : {business_threshold:.2f}"
    )
    print(
        f"Accuracy      : {business_row['accuracy']:.4f}"
    )
    print(
        f"Precision     : {business_row['precision']:.4f}"
    )
    print(
        f"Recall        : {business_row['recall']:.4f}"
    )
    print(
        f"F1 Score      : {business_row['f1_score']:.4f}"
    )
    print(
        f"TN            : {int(business_row['tn'])}"
    )
    print(
        f"FP            : {int(business_row['fp'])}"
    )
    print(
        f"FN            : {int(business_row['fn'])}"
    )
    print(
        f"TP            : {int(business_row['tp'])}"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_rows = [
        {
            "strategy": "default",
            "threshold": DEFAULT_THRESHOLD,
            **{
                k: v
                for k, v in default_metrics.items()
                if k != "threshold"
            },
        },
        {
            "strategy": "best_f1",
            **best_f1_row.to_dict(),
        },
        {
            "strategy": "business_recall_0.50",
            **business_row.to_dict(),
        },
    ]

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    results_df.to_csv(
        THRESHOLD_RESULTS_FILE,
        index=False,
    )

    comparison_df.to_csv(
        THRESHOLD_COMPARISON_FILE,
        index=False,
    )

    metadata = {
        "project": (
            "Enterprise Banking Risk "
            "Intelligence Platform"
        ),
        "pipeline": "fraud_threshold_optimization",
        "target_column": target_column,
        "validation_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "model_type": type(model).__name__,
        "model_file": str(MODEL_FILE),
        "validation_file": str(VALIDATION_FILE),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "default_threshold": DEFAULT_THRESHOLD,
        "best_f1_threshold": best_f1_threshold,
        "business_threshold": business_threshold,
        "business_recall_target": 0.50,
    }

    with open(
        THRESHOLD_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FRAUD THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(
        f"\nBest F1 threshold       : "
        f"{best_f1_threshold:.2f}"
    )

    print(
        f"Business threshold      : "
        f"{business_threshold:.2f}"
    )

    print(
        f"ROC-AUC                 : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC                  : "
        f"{pr_auc:.4f}"
    )

    print("\nArtifacts:")

    print(
        f"Threshold results       : "
        f"{THRESHOLD_RESULTS_FILE}"
    )

    print(
        f"Threshold comparison    : "
        f"{THRESHOLD_COMPARISON_FILE}"
    )

    print(
        f"Threshold metadata      : "
        f"{THRESHOLD_METADATA_FILE}"
    )

    print(
        "\nFraud threshold optimization "
        "pipeline completed successfully."
    )


if __name__ == "__main__":
    main()