from pathlib import Path
import json
import joblib
import warnings

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# FRAUD ML MODEL TRAINING PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud"
)

OUTPUT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "fraud"
)

OUTPUT_REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
)

OUTPUT_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_FILE = (
    DATA_DIR
    / "fraud_train_processed.csv"
)

VALIDATION_FILE = (
    DATA_DIR
    / "fraud_validation_processed.csv"
)

PREPROCESSOR_FILE = (
    DATA_DIR
    / "fraud_preprocessor.joblib"
)

TARGET_COLUMN = "SeriousDlqin2yrs"

RANDOM_STATE = 42


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL_FILE = (
    OUTPUT_MODEL_DIR
    / "fraud_best_model.joblib"
)

LOGISTIC_MODEL_FILE = (
    OUTPUT_MODEL_DIR
    / "fraud_logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_FILE = (
    OUTPUT_MODEL_DIR
    / "fraud_random_forest.joblib"
)

GRADIENT_BOOSTING_MODEL_FILE = (
    OUTPUT_MODEL_DIR
    / "fraud_gradient_boosting.joblib"
)

MODEL_COMPARISON_FILE = (
    OUTPUT_REPORT_DIR
    / "fraud_model_comparison.csv"
)

MODEL_EVALUATION_FILE = (
    OUTPUT_REPORT_DIR
    / "fraud_model_evaluation.json"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path, description):

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def evaluate_model(model, X_valid, y_valid):

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_valid,
        predictions,
    )

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        predictions,
        labels=[0, 1],
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FRAUD ML MODEL TRAINING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Required artifacts
    # --------------------------------------------------------

    print("\nChecking required artifacts...")

    require_file(
        TRAIN_FILE,
        "Training dataset",
    )

    require_file(
        VALIDATION_FILE,
        "Validation dataset",
    )

    require_file(
        PREPROCESSOR_FILE,
        "Preprocessor",
    )

    print("All required artifacts found.")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    valid_df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Training rows     : {len(train_df):,}"
    )

    print(
        f"Validation rows   : {len(valid_df):,}"
    )

    print(
        f"Training columns  : {len(train_df.columns)}"
    )

    print(
        f"Validation columns: {len(valid_df.columns)}"
    )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column missing from training data: "
            f"{TARGET_COLUMN}"
        )

    if TARGET_COLUMN not in valid_df.columns:
        raise ValueError(
            f"Target column missing from validation data: "
            f"{TARGET_COLUMN}"
        )

    # --------------------------------------------------------
    # Separate X / y
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_valid = valid_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_valid = valid_df[
        TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # Feature alignment
    # --------------------------------------------------------

    print("\nFeature validation")

    print("-" * 70)

    print(
        f"Feature count: {X_train.shape[1]}"
    )

    if list(X_train.columns) != list(
        X_valid.columns
    ):

        raise ValueError(
            "Training and validation features "
            "are not aligned."
        )

    print(
        "Training and validation features: ALIGNED"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution")

    print("\nTraining:")

    print(
        y_train
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidation:")

    print(
        y_valid
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Load preprocessor
    # --------------------------------------------------------

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        f"Preprocessor type: "
        f"{type(preprocessor).__name__}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "gradient_boosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),
    }

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    evaluation_details = {}

    trained_models = {}

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "=" * 70)

        print(
            f"TRAINING: "
            f"{model_name.upper()}"
        )

        print("=" * 70)

        print("\nFitting model...")

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Model training complete."
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        print(
            "\nEvaluating on validation dataset..."
        )

        metrics = evaluate_model(
            model,
            X_valid,
            y_valid,
        )

        print("\nValidation Metrics")

        print(
            f"Accuracy  : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{metrics['pr_auc']:.4f}"
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

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        evaluation_details[
            model_name
        ] = metrics

        trained_models[
            model_name
        ] = model

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Best model
    #
    # PR-AUC is primary because fraud/default
    # data is imbalanced.
    # --------------------------------------------------------

    comparison_df = comparison_df.sort_values(
        by=[
            "pr_auc",
            "roc_auc",
            "f1_score",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    best_model_name = (
        comparison_df.iloc[0]["model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_metrics = evaluation_details[
        best_model_name
    ]

    # ========================================================
    # SAVE INDIVIDUAL MODELS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "SAVING MODEL ARTIFACTS"
    )

    print("=" * 70)

    joblib.dump(
        trained_models[
            "logistic_regression"
        ],
        LOGISTIC_MODEL_FILE,
    )

    joblib.dump(
        trained_models[
            "random_forest"
        ],
        RANDOM_FOREST_MODEL_FILE,
    )

    joblib.dump(
        trained_models[
            "gradient_boosting"
        ],
        GRADIENT_BOOSTING_MODEL_FILE,
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        BEST_MODEL_FILE,
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison_df.to_csv(
        MODEL_COMPARISON_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Evaluation metadata
    # --------------------------------------------------------

    evaluation_report = {

        "pipeline": (
            "Fraud ML Model Training Pipeline"
        ),

        "project_root": str(
            PROJECT_ROOT
        ),

        "training_file": str(
            TRAIN_FILE
        ),

        "validation_file": str(
            VALIDATION_FILE
        ),

        "preprocessor_file": str(
            PREPROCESSOR_FILE
        ),

        "target_column": TARGET_COLUMN,

        "random_state": RANDOM_STATE,

        "training_rows": int(
            len(train_df)
        ),

        "validation_rows": int(
            len(valid_df)
        ),

        "feature_count": int(
            X_train.shape[1]
        ),

        "models": evaluation_details,

        "model_selection_metric": (
            "PR-AUC"
        ),

        "best_model": best_model_name,

        "best_model_metrics": best_metrics,

        "artifacts": {

            "best_model": str(
                BEST_MODEL_FILE
            ),

            "logistic_regression": str(
                LOGISTIC_MODEL_FILE
            ),

            "random_forest": str(
                RANDOM_FOREST_MODEL_FILE
            ),

            "gradient_boosting": str(
                GRADIENT_BOOSTING_MODEL_FILE
            ),

            "comparison_report": str(
                MODEL_COMPARISON_FILE
            ),

            "evaluation_report": str(
                MODEL_EVALUATION_FILE
            ),
        },
    }

    with MODEL_EVALUATION_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation_report,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)

    print(
        f"Best model : "
        f"{best_model_name}"
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

    print(
        f"\nBest model saved:"
    )

    print(
        BEST_MODEL_FILE
    )

    print(
        "\nIndividual model artifacts saved."
    )

    print(
        f"\nComparison report:"
    )

    print(
        MODEL_COMPARISON_FILE
    )

    print(
        f"\nEvaluation report:"
    )

    print(
        MODEL_EVALUATION_FILE
    )

    print(
        "\nFraud model training pipeline "
        "completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()