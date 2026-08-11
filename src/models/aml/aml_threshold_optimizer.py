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
# AML THRESHOLD OPTIMIZATION PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml"
    / "aml_validation_processed.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "aml"
)

MODEL_FILE = MODEL_DIR / "aml_logistic_regression.joblib"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_threshold"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "aml_threshold_results.csv"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "aml_threshold_comparison.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "aml_threshold_metadata.json"
)

TARGET = "is_laundering"


def evaluate_threshold(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
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
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():

    print("# AML THRESHOLD OPTIMIZATION PIPELINE")
    print()

    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print()

    # --------------------------------------------------------
    # Check artifacts
    # --------------------------------------------------------

    print("Checking required artifacts...")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"AML model not found:\n{MODEL_FILE}"
        )

    print("All required artifacts found.")
    print()

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("Loading validation dataset...")

    df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Validation rows    : {len(df):,}"
    )

    print(
        f"Validation columns : {len(df.columns)}"
    )

    print()

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET].astype(int)

    print("Target distribution:")
    print(
        y.value_counts().to_string()
    )
    print()

    positive_count = int(y.sum())

    if positive_count < 2:

        print(
            "WARNING: Validation dataset contains "
            f"only {positive_count} AML positive case."
        )

        print(
            "Threshold metrics will be unstable "
            "and should NOT be treated as production-grade."
        )

        print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading AML model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type: {type(model).__name__}"
    )

    print()

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print(
        "Generating AML prediction probabilities..."
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    print(
        f"Minimum probability: {probabilities.min():.6f}"
    )

    print(
        f"Maximum probability: {probabilities.max():.6f}"
    )

    print()

    # --------------------------------------------------------
    # Default threshold
    # --------------------------------------------------------

    default_result = evaluate_threshold(
        y,
        probabilities,
        0.50
    )

    print("## Default Threshold: 0.50")
    print()

    print(
        f"Accuracy  : {default_result['accuracy']:.4f}"
    )

    print(
        f"Precision : {default_result['precision']:.4f}"
    )

    print(
        f"Recall    : {default_result['recall']:.4f}"
    )

    print(
        f"F1 Score  : {default_result['f1']:.4f}"
    )

    print(
        f"TN        : {default_result['tn']}"
    )

    print(
        f"FP        : {default_result['fp']}"
    )

    print(
        f"FN        : {default_result['fn']}"
    )

    print(
        f"TP        : {default_result['tp']}"
    )

    print()

    # --------------------------------------------------------
    # ROC / PR metrics
    # --------------------------------------------------------

    if y.nunique() == 2:

        roc_auc = roc_auc_score(
            y,
            probabilities
        )

        pr_auc = average_precision_score(
            y,
            probabilities
        )

    else:

        roc_auc = None
        pr_auc = None

    # --------------------------------------------------------
    # Threshold search
    # --------------------------------------------------------

    print(
        "Searching classification thresholds..."
    )
    print()

    thresholds = np.arange(
        0.01,
        1.00,
        0.01
    )

    results = []

    for threshold in thresholds:

        result = evaluate_threshold(
            y,
            probabilities,
            threshold
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Best F1 threshold
    # --------------------------------------------------------

    best_f1_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_f1_threshold = float(
        best_f1_row["threshold"]
    )

    # --------------------------------------------------------
    # Business threshold
    # --------------------------------------------------------
    #
    # AML systems generally prioritize recall.
    # With only one validation positive, however,
    # this threshold is experimental only.
    #

    recall_candidates = results_df[
        results_df["recall"] >= 0.90
    ]

    if not recall_candidates.empty:

        business_row = (
            recall_candidates
            .sort_values(
                ["precision", "f1"],
                ascending=False
            )
            .iloc[0]
        )

    else:

        business_row = best_f1_row

    business_threshold = float(
        business_row["threshold"]
    )

    print(
        f"ROC-AUC             : "
        f"{roc_auc if roc_auc is not None else 'N/A'}"
    )

    print(
        f"PR-AUC              : "
        f"{pr_auc if pr_auc is not None else 'N/A'}"
    )

    print(
        "Default threshold    : 0.50"
    )

    print(
        f"Best F1 threshold   : "
        f"{best_f1_threshold:.2f}"
    )

    print(
        f"Business threshold  : "
        f"{business_threshold:.2f}"
    )

    print()

    print("Best F1 threshold metrics:")

    print(
        f"Accuracy  : "
        f"{best_f1_row['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_f1_row['f1']:.4f}"
    )

    print()

    print("Business threshold metrics:")

    print(
        f"Accuracy  : "
        f"{business_row['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{business_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{business_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{business_row['f1']:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    comparison_rows = [
        {
            "strategy": "default",
            **default_result
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

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        COMPARISON_FILE,
        index=False
    )

    metadata = {
        "pipeline": "AML threshold optimization",

        "validation_file": str(
            VALIDATION_FILE
        ),

        "model_file": str(
            MODEL_FILE
        ),

        "validation_rows": int(
            len(df)
        ),

        "validation_positive_cases": int(
            y.sum()
        ),

        "validation_negative_cases": int(
            (y == 0).sum()
        ),

        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),

        "pr_auc": (
            float(pr_auc)
            if pr_auc is not None
            else None
        ),

        "default_threshold": 0.50,

        "best_f1_threshold": best_f1_threshold,

        "business_threshold": business_threshold,

        "warning": (
            "Validation contains only one positive AML case. "
            "Threshold metrics are highly unstable and "
            "must not be treated as production-grade."
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("## Output artifacts")
    print()

    print(
        f"Threshold results     : {RESULTS_FILE}"
    )

    print(
        f"Threshold comparison  : {COMPARISON_FILE}"
    )

    print(
        f"Threshold metadata    : {METADATA_FILE}"
    )

    print()

    print(
        "AML threshold optimization "
        "pipeline completed successfully."
    )


if __name__ == "__main__":
    main()