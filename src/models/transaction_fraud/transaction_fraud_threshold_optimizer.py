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

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "transaction_fraud"
MODEL_DIR = PROJECT_ROOT / "models" / "transaction_fraud"
OUTPUT_DIR = PROJECT_ROOT / "data" / "staging" / "transaction_fraud"

VALIDATION_FILE = DATA_DIR / "transaction_fraud_validation_processed.csv"
BEST_MODEL_FILE = MODEL_DIR / "transaction_fraud_best_model.joblib"

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

MIN_RECALL = 0.70


def main():

    print("=" * 70)
    print("TRANSACTION FRAUD THRESHOLD OPTIMIZATION")
    print("=" * 70)

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_FILE}"
        )

    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best model not found:\n{BEST_MODEL_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading validation dataset...")
    df = pd.read_csv(VALIDATION_FILE)

    target = "Class"

    if target not in df.columns:
        raise ValueError(f"Target column missing: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    print(f"Validation rows : {len(df):,}")
    print(f"Features        : {len(X.columns)}")

    print("\nLoading best model...")
    model = joblib.load(BEST_MODEL_FILE)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba().")

    probabilities = model.predict_proba(X)[:, 1]

    print(
        f"Probability range: "
        f"{probabilities.min():.6f} - {probabilities.max():.6f}"
    )

    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    results = []

    print("\nThreshold analysis")
    print("-" * 70)

    for threshold in THRESHOLDS:

        predictions = (probabilities >= threshold).astype(int)

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(
            y, predictions, zero_division=0
        )
        recall = recall_score(
            y, predictions, zero_division=0
        )
        f1 = f1_score(
            y, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append({
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
            "predicted_fraud": int(predictions.sum()),
        })

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"Predicted Fraud={predictions.sum()}"
        )

    results_df = pd.DataFrame(results)

    report_file = (
        OUTPUT_DIR /
        "transaction_fraud_threshold_analysis.csv"
    )

    results_df.to_csv(
        report_file,
        index=False,
    )

    eligible = results_df[
        results_df["recall"] >= MIN_RECALL
    ]

    if len(eligible) > 0:

        selected = eligible.sort_values(
            by=["f1_score", "precision"],
            ascending=False,
        ).iloc[0]

        reason = (
            "Highest F1 among thresholds "
            f"with recall >= {MIN_RECALL:.2f}"
        )

    else:

        selected = results_df.sort_values(
            by=["recall", "f1_score"],
            ascending=False,
        ).iloc[0]

        reason = (
            "No threshold satisfied minimum recall; "
            "selected highest recall."
        )

    selected_threshold = float(
        selected["threshold"]
    )

    metadata = {
        "model_file": str(BEST_MODEL_FILE),
        "validation_file": str(VALIDATION_FILE),
        "target_column": target,
        "validation_rows": int(len(df)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "minimum_recall_requirement": MIN_RECALL,
        "selected_threshold": selected_threshold,
        "selected_precision": float(selected["precision"]),
        "selected_recall": float(selected["recall"]),
        "selected_f1_score": float(selected["f1_score"]),
        "selected_true_positive": int(selected["true_positive"]),
        "selected_false_positive": int(selected["false_positive"]),
        "selected_false_negative": int(selected["false_negative"]),
        "selection_reason": reason,
    }

    metadata_file = (
        OUTPUT_DIR /
        "transaction_fraud_threshold_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("SELECTED OPERATING THRESHOLD")
    print("=" * 70)

    print(f"Threshold : {selected_threshold:.2f}")
    print(f"Precision : {selected['precision']:.4f}")
    print(f"Recall    : {selected['recall']:.4f}")
    print(f"F1 Score  : {selected['f1_score']:.4f}")
    print(f"TP        : {int(selected['true_positive'])}")
    print(f"FP        : {int(selected['false_positive'])}")
    print(f"FN        : {int(selected['false_negative'])}")
    print(f"Reason    : {reason}")

    print("\nReports saved:")
    print(report_file)
    print(metadata_file)


if __name__ == "__main__":
    main()
