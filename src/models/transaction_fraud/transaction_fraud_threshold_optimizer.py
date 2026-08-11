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


# ============================================================
# TRANSACTION FRAUD THRESHOLD OPTIMIZATION PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transaction_fraud"
    / "transaction_fraud_validation_processed.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "transaction_fraud"
    / "transaction_fraud_best_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "transaction_fraud"
)

THRESHOLD_RESULTS_FILE = (
    OUTPUT_DIR
    / "transaction_fraud_threshold_results.csv"
)

THRESHOLD_COMPARISON_FILE = (
    OUTPUT_DIR
    / "transaction_fraud_threshold_comparison.csv"
)

THRESHOLD_METADATA_FILE = (
    OUTPUT_DIR
    / "transaction_fraud_threshold_metadata.json"
)

TARGET_COLUMN = "Class"


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRANSACTION FRAUD THRESHOLD OPTIMIZATION PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # ========================================================
    # 1. Check artifacts
    # ========================================================

    print("\nChecking required artifacts...")

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_FILE}"
        )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Best model not found:\n{MODEL_FILE}"
        )

    print("All required artifacts found.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nValidation dataset:")
    print(VALIDATION_FILE)

    print("\nBest model:")
    print(MODEL_FILE)

    # ========================================================
    # 2. Load validation dataset
    # ========================================================

    print("\nLoading validation dataset...")

    df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Validation rows    : {len(df):,}"
    )

    print(
        f"Validation columns : {len(df.columns)}"
    )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN].astype(int)

    print(
        f"Validation features: {X.shape[1]}"
    )

    print("\nTarget distribution")

    print(
        y.value_counts()
        .sort_index()
    )

    # ========================================================
    # 3. Load model
    # ========================================================

    print("\nLoading best model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type        : "
        f"{type(model).__name__}"
    )

    # ========================================================
    # 4. Generate probabilities
    # ========================================================

    print("\nGenerating prediction probabilities...")

    probabilities = model.predict_proba(
        X
    )[:, 1]

    print(
        f"Minimum probability: "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Maximum probability: "
        f"{probabilities.max():.6f}"
    )

    # ========================================================
    # 5. Default threshold
    # ========================================================

    DEFAULT_THRESHOLD = 0.50

    print("\n" + "=" * 70)
    print("DEFAULT THRESHOLD = 0.50")
    print("=" * 70)

    default_metrics = calculate_metrics(
        y,
        probabilities,
        DEFAULT_THRESHOLD
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

    # ========================================================
    # 6. Search thresholds
    # ========================================================

    print("\nSearching classification thresholds...")

    thresholds = np.arange(
        0.01,
        1.00,
        0.01
    )

    threshold_results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y,
            probabilities,
            threshold
        )

        threshold_results.append(
            metrics
        )

    results_df = pd.DataFrame(
        threshold_results
    )

    # ========================================================
    # 7. Best F1 threshold
    # ========================================================

    best_f1_row = results_df.loc[
        results_df["f1_score"].idxmax()
    ]

    best_f1_threshold = float(
        best_f1_row["threshold"]
    )

    print("\n" + "=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)

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

    # ========================================================
    # 8. Business-oriented threshold
    # ========================================================
    #
    # Fraud detection is highly imbalanced.
    # Missing fraud can be more costly than reviewing
    # additional legitimate transactions.
    #
    # We therefore select the highest threshold that
    # achieves recall >= 0.70.
    #
    # ========================================================

    RECALL_TARGET = 0.70

    eligible = results_df[
        results_df["recall"] >= RECALL_TARGET
    ].copy()

    if eligible.empty:

        business_row = results_df.loc[
            results_df["recall"].idxmax()
        ]

    else:

        business_row = eligible.sort_values(
            by=[
                "precision",
                "threshold"
            ],
            ascending=[
                False,
                True
            ]
        ).iloc[0]

    business_threshold = float(
        business_row["threshold"]
    )

    print("\n" + "=" * 70)
    print("BUSINESS-ORIENTED THRESHOLD")
    print("=" * 70)

    print(
        f"Recall target : >= {RECALL_TARGET:.2f}"
    )

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

    # ========================================================
    # 9. ROC-AUC / PR-AUC
    # ========================================================

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    print("\n" + "=" * 70)
    print("MODEL-LEVEL METRICS")
    print("=" * 70)

    print(
        f"ROC-AUC                 : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC                  : {pr_auc:.4f}"
    )

    print(
        f"Default threshold       : {DEFAULT_THRESHOLD:.2f}"
    )

    print(
        f"Best F1 threshold       : {best_f1_threshold:.2f}"
    )

    print(
        f"Business threshold      : {business_threshold:.2f}"
    )

    # ========================================================
    # 10. Save complete threshold results
    # ========================================================

    results_df.to_csv(
        THRESHOLD_RESULTS_FILE,
        index=False
    )

    # ========================================================
    # 11. Save comparison
    # ========================================================

    comparison = pd.DataFrame(
        [
            {
                "strategy": "default",
                **default_metrics
            },
            {
                "strategy": "best_f1",
                **best_f1_row.to_dict()
            },
            {
                "strategy": "business",
                **business_row.to_dict()
            },
        ]
    )

    comparison.to_csv(
        THRESHOLD_COMPARISON_FILE,
        index=False
    )

    # ========================================================
    # 12. Save metadata
    # ========================================================

    metadata = {

        "pipeline":
            "transaction_fraud_threshold_optimization",

        "model_file":
            str(MODEL_FILE),

        "validation_file":
            str(VALIDATION_FILE),

        "validation_rows":
            int(len(df)),

        "fraud_cases":
            int(y.sum()),

        "legitimate_cases":
            int((y == 0).sum()),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "default_threshold":
            float(DEFAULT_THRESHOLD),

        "best_f1_threshold":
            float(best_f1_threshold),

        "business_threshold":
            float(business_threshold),

        "business_recall_target":
            float(RECALL_TARGET),

        "selection_method":
            "best F1 for model optimization; "
            "business threshold selected using "
            "precision among thresholds meeting "
            "minimum recall target",

        "default_metrics":
            default_metrics,

        "best_f1_metrics":
            best_f1_row.to_dict(),

        "business_metrics":
            business_row.to_dict(),
    }

    with open(
        THRESHOLD_METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # ========================================================
    # 13. Final output
    # ========================================================

    print("\n" + "=" * 70)
    print("Artifacts")
    print("=" * 70)

    print(
        f"\nThreshold results       : "
        f"{THRESHOLD_RESULTS_FILE}"
    )

    print(
        f"Threshold comparison   : "
        f"{THRESHOLD_COMPARISON_FILE}"
    )

    print(
        f"Threshold metadata     : "
        f"{THRESHOLD_METADATA_FILE}"
    )

    print("\n" + "=" * 70)

    print(
        "Transaction fraud threshold "
        "optimization pipeline completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()