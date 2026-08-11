from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# TRANSACTION FRAUD ML PREPROCESSING PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "transaction_fraud"
    / "transaction_fraud_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transaction_fraud"
)

TRAIN_OUTPUT = OUTPUT_DIR / "transaction_fraud_train_processed.csv"
VALIDATION_OUTPUT = OUTPUT_DIR / "transaction_fraud_validation_processed.csv"
PREPROCESSOR_OUTPUT = OUTPUT_DIR / "transaction_fraud_preprocessor.joblib"
METADATA_OUTPUT = OUTPUT_DIR / "transaction_fraud_preprocessing_metadata.json"

TARGET_COLUMN = "Class"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def main():
    print("=" * 70)
    print("TRANSACTION FRAUD ML PREPROCESSING PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # 1. Check input
    # --------------------------------------------------------

    print("\nChecking input dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    print(f"Input file: {INPUT_FILE}")

    # --------------------------------------------------------
    # 2. Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 3. Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows    : {len(df):,}")
    print(f"Input columns : {len(df.columns)}")

    # --------------------------------------------------------
    # 4. Validate target
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    print(f"\nTarget column : {TARGET_COLUMN}")

    # --------------------------------------------------------
    # 5. Separate features and target
    # --------------------------------------------------------

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    print(f"Feature count : {X.shape[1]}")

    # --------------------------------------------------------
    # 6. Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution")

    target_distribution = y.value_counts().sort_index()

    for value, count in target_distribution.items():
        percentage = count / len(y) * 100
        label = "Legitimate" if value == 0 else "Fraud"

        print(
            f"{label:<12}: {count:>10,} "
            f"({percentage:.6f}%)"
        )

    # --------------------------------------------------------
    # 7. Feature classification
    # --------------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\n## Feature classification")

    print(
        f"Numeric features     : {len(numeric_features)}"
    )

    print(
        f"Categorical features : {len(categorical_features)}"
    )

    if numeric_features:
        print("\nNumeric columns:")
        print(numeric_features)

    if categorical_features:
        print("\nCategorical columns:")
        print(categorical_features)

    # --------------------------------------------------------
    # 8. Train / validation split
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n## Dataset split")

    print(
        f"Training rows   : {len(X_train):,}"
    )

    print(
        f"Validation rows : {len(X_validation):,}"
    )

    print("\nTarget distribution")

    print("\nTraining:")
    print(y_train.value_counts().sort_index())

    print("\nValidation:")
    print(y_validation.value_counts().sort_index())

    # --------------------------------------------------------
    # 9. Build preprocessing pipeline
    # --------------------------------------------------------

    print("\nBuilding preprocessing pipeline...")

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # --------------------------------------------------------
    # 10. Fit ONLY on training data
    # --------------------------------------------------------

    print(
        "\nFitting preprocessing pipeline "
        "on training data only..."
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_validation_processed = preprocessor.transform(
        X_validation
    )

    # --------------------------------------------------------
    # 11. Feature names
    # --------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    feature_count = len(feature_names)

    print("\n## Processed datasets")

    print(
        f"Training matrix   : {X_train_processed.shape}"
    )

    print(
        f"Validation matrix : {X_validation_processed.shape}"
    )

    print(
        f"Final feature count: {feature_count}"
    )

    # --------------------------------------------------------
    # 12. Validate matrices
    # --------------------------------------------------------

    if np.isnan(X_train_processed).any():
        raise ValueError(
            "NaN values found in processed training data."
        )

    if np.isnan(X_validation_processed).any():
        raise ValueError(
            "NaN values found in processed validation data."
        )

    if np.isinf(X_train_processed).any():
        raise ValueError(
            "Infinite values found in processed training data."
        )

    if np.isinf(X_validation_processed).any():
        raise ValueError(
            "Infinite values found in processed validation data."
        )

    if X_train_processed.shape[1] != X_validation_processed.shape[1]:
        raise ValueError(
            "Training and validation feature counts do not match."
        )

    # --------------------------------------------------------
    # 13. Create output DataFrames
    # --------------------------------------------------------

    train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    train_processed_df[TARGET_COLUMN] = y_train.values

    validation_processed_df = pd.DataFrame(
        X_validation_processed,
        columns=feature_names,
        index=X_validation.index,
    )

    validation_processed_df[TARGET_COLUMN] = y_validation.values

    # --------------------------------------------------------
    # 14. Save processed datasets
    # --------------------------------------------------------

    print("\nSaving processed datasets...")

    train_processed_df.to_csv(
        TRAIN_OUTPUT,
        index=False,
    )

    validation_processed_df.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # 15. Save preprocessor
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        PREPROCESSOR_OUTPUT,
    )

    # --------------------------------------------------------
    # 16. Metadata
    # --------------------------------------------------------

    metadata = {
        "pipeline": "transaction_fraud_ml_preprocessing",
        "input_file": str(INPUT_FILE),
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(
            len(categorical_features)
        ),
        "final_feature_count": int(feature_count),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": feature_names.tolist(),
        "training_target_distribution": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "validation_target_distribution": {
            str(k): int(v)
            for k, v in y_validation.value_counts().sort_index().items()
        },
        "preprocessor_type": type(preprocessor).__name__,
        "scaler": "StandardScaler",
        "categorical_encoder": (
            "OneHotEncoder"
            if categorical_features
            else None
        ),
        "fit_scope": "training_data_only",
    }

    with open(
        METADATA_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # 17. Verify outputs
    # --------------------------------------------------------

    print("\nVerifying outputs...")

    required_outputs = [
        TRAIN_OUTPUT,
        VALIDATION_OUTPUT,
        PREPROCESSOR_OUTPUT,
        METADATA_OUTPUT,
    ]

    for output in required_outputs:
        if not output.exists():
            raise FileNotFoundError(
                f"Expected output was not created:\n{output}"
            )

    # Reload and verify processed files
    train_check = pd.read_csv(TRAIN_OUTPUT)
    validation_check = pd.read_csv(VALIDATION_OUTPUT)

    expected_train_columns = feature_count + 1
    expected_validation_columns = feature_count + 1

    if train_check.shape[1] != expected_train_columns:
        raise ValueError(
            "Training output column count verification failed."
        )

    if validation_check.shape[1] != expected_validation_columns:
        raise ValueError(
            "Validation output column count verification failed."
        )

    if train_check[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Missing target values in training output."
        )

    if validation_check[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Missing target values in validation output."
        )

    print("\n## Output artifacts")

    print(
        f"Training output   : {TRAIN_OUTPUT}"
    )

    print(
        f"Validation output : {VALIDATION_OUTPUT}"
    )

    print(
        f"Preprocessor      : {PREPROCESSOR_OUTPUT}"
    )

    print(
        f"Metadata          : {METADATA_OUTPUT}"
    )

    print(
        f"\nFinal feature count: {feature_count}"
    )

    print("\n" + "=" * 70)
    print(
        "Transaction fraud ML preprocessing "
        "pipeline completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()