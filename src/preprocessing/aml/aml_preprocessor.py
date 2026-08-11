from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# AML PREPROCESSING PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml"
    / "aml_train"
    / "aml_train_features.csv"
)

VALIDATION_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml"
    / "aml_validation"
    / "aml_validation_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml"
)

TRAIN_FILE = OUTPUT_DIR / "aml_train_processed.csv"
VALIDATION_FILE = OUTPUT_DIR / "aml_validation_processed.csv"
PREPROCESSOR_FILE = OUTPUT_DIR / "aml_preprocessor.joblib"
METADATA_FILE = OUTPUT_DIR / "aml_preprocessing_metadata.json"

TARGET = "is_laundering"


def main():

    print("# AML PREPROCESSING PIPELINE")
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print()

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    print("Checking input datasets...")

    if not TRAIN_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_INPUT_FILE}"
        )

    if not VALIDATION_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_INPUT_FILE}"
        )

    print(f"Training input   : {TRAIN_INPUT_FILE}")
    print(f"Validation input : {VALIDATION_INPUT_FILE}")
    print()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("Loading AML train dataset...")

    train_df = pd.read_csv(TRAIN_INPUT_FILE)

    print(f"Training rows    : {len(train_df):,}")
    print(f"Training columns : {len(train_df.columns)}")
    print()

    print("Loading AML validation dataset...")

    validation_df = pd.read_csv(VALIDATION_INPUT_FILE)

    print(f"Validation rows    : {len(validation_df):,}")
    print(f"Validation columns : {len(validation_df.columns)}")
    print()

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if TARGET not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in training dataset."
        )

    if TARGET not in validation_df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in validation dataset."
        )

    print("Training target distribution:")
    print(train_df[TARGET].value_counts().to_string())
    print()

    print("Validation target distribution:")
    print(validation_df[TARGET].value_counts().to_string())
    print()

    # --------------------------------------------------------
    # Remove identifier columns
    # --------------------------------------------------------

    identifier_columns = [
        "record_key",
        "from_account",
        "to_account",
        "from_bank",
        "to_bank",
    ]

    identifier_columns = [
        col
        for col in identifier_columns
        if col in train_df.columns
    ]

    print("Removing high-cardinality identifier columns...")
    print(identifier_columns)
    print()

    X_train = train_df.drop(
        columns=[TARGET] + identifier_columns
    )

    y_train = train_df[TARGET].astype(int)

    X_validation = validation_df.drop(
        columns=[TARGET] + identifier_columns
    )

    y_validation = validation_df[TARGET].astype(int)

    # --------------------------------------------------------
    # Feature alignment
    # --------------------------------------------------------

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns are not aligned."
        )

    print("Training and validation features: ALIGNED")
    print()

    # --------------------------------------------------------
    # Detect feature types
    # --------------------------------------------------------

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_features = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print("## Feature classification")
    print()
    print(f"Numeric features     : {len(numeric_features)}")
    print(f"Categorical features : {len(categorical_features)}")
    print()

    if categorical_features:
        print("Categorical columns:")
        print(categorical_features)
        print()

    # --------------------------------------------------------
    # Build preprocessing pipeline
    # --------------------------------------------------------

    print("Building preprocessing pipeline...")
    print()

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
    # Fit ONLY on training data
    # --------------------------------------------------------

    print(
        "Fitting preprocessing pipeline on training data only..."
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_validation_processed = preprocessor.transform(
        X_validation
    )

    print("Preprocessing complete.")
    print()

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    print("## Processed datasets")
    print()

    print(
        f"Training matrix   : {X_train_processed.shape}"
    )

    print(
        f"Validation matrix : {X_validation_processed.shape}"
    )

    print(
        f"Final feature count: {len(feature_names)}"
    )

    print()

    # --------------------------------------------------------
    # Create output DataFrames
    # --------------------------------------------------------

    train_output = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
    )

    validation_output = pd.DataFrame(
        X_validation_processed,
        columns=feature_names,
    )

    train_output[TARGET] = y_train.to_numpy()

    validation_output[TARGET] = y_validation.to_numpy()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save processed datasets
    # --------------------------------------------------------

    print("Saving processed datasets...")

    train_output.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation_output.to_csv(
        VALIDATION_FILE,
        index=False
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "pipeline": "AML preprocessing",
        "training_input_file": str(TRAIN_INPUT_FILE),
        "validation_input_file": str(VALIDATION_INPUT_FILE),
        "target": TARGET,

        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),

        "input_features": int(len(X_train.columns)),
        "final_features": int(len(feature_names)),

        "numeric_features": numeric_features,
        "categorical_features": categorical_features,

        "removed_identifier_columns": identifier_columns,

        "training_positive_cases": int(y_train.sum()),
        "training_negative_cases": int(
            (y_train == 0).sum()
        ),

        "validation_positive_cases": int(y_validation.sum()),
        "validation_negative_cases": int(
            (y_validation == 0).sum()
        ),

        "feature_names": feature_names,

        "random_state": 42,

        "scaling": "StandardScaler",

        "categorical_encoding": (
            "OneHotEncoder(handle_unknown='ignore')"
        ),

        "fit_on_training_only": True,

        "split_strategy": (
            "Pre-created stratified AML split "
            "with all positive AML cases in training"
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
    # Verify outputs
    # --------------------------------------------------------

    print()
    print("Verifying outputs...")

    for file_path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        PREPROCESSOR_FILE,
        METADATA_FILE,
    ]:

        if not file_path.exists():
            raise RuntimeError(
                f"Expected output not created: {file_path}"
            )

    print("All output artifacts verified.")
    print()

    # --------------------------------------------------------
    # Output artifacts
    # --------------------------------------------------------

    print("## Output artifacts")
    print()

    print(
        f"Training output   : {TRAIN_FILE}"
    )

    print(
        f"Validation output : {VALIDATION_FILE}"
    )

    print(
        f"Preprocessor      : {PREPROCESSOR_FILE}"
    )

    print(
        f"Metadata          : {METADATA_FILE}"
    )

    print()

    print(
        f"Final feature count: {len(feature_names)}"
    )

    print()

    print(
        "AML preprocessing pipeline completed successfully."
    )


if __name__ == "__main__":
    main()