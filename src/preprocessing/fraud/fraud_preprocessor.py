from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud"
    / "fraud_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "SeriousDlqin2yrs"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(numeric_columns, categorical_columns):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FRAUD ML PREPROCESSING PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    print("\nChecking input dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nFraud feature dataset not found:\n{INPUT_FILE}"
        )

    print(f"Input file: {INPUT_FILE}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows    : {len(df):,}")
    print(f"Input columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET_COLUMN}"
        )

    # --------------------------------------------------------
    # Separate target and features
    # --------------------------------------------------------

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Remove identifier
    # --------------------------------------------------------

    if "Unnamed: 0" in X.columns:
        X = X.drop(columns=["Unnamed: 0"])

    # --------------------------------------------------------
    # Identify categorical columns
    # --------------------------------------------------------

    categorical_columns = [
        column
        for column in X.columns
        if (
            pd.api.types.is_object_dtype(X[column])
            or pd.api.types.is_string_dtype(X[column])
            or isinstance(
                X[column].dtype,
                pd.CategoricalDtype,
            )
        )
    ]

    # --------------------------------------------------------
    # Identify numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    # --------------------------------------------------------
    # Feature classification
    # --------------------------------------------------------

    print("\nFeature classification")
    print("-" * 70)

    print(
        f"Numeric features     : "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_columns)}"
    )

    print("\nNumeric columns:")
    print(numeric_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDataset split")
    print("-" * 70)

    print(
        f"Training rows   : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows : "
        f"{len(X_valid):,}"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution")

    print("\nTraining:")
    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidation:")
    print(
        y_valid.value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Build preprocessor
    # --------------------------------------------------------

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    # --------------------------------------------------------
    # Fit ONLY on training data
    # --------------------------------------------------------

    print(
        "\nFitting preprocessing pipeline "
        "on training data only..."
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_valid_processed = preprocessor.transform(
        X_valid
    )

    # --------------------------------------------------------
    # Processed shapes
    # --------------------------------------------------------

    print("\nProcessed datasets")
    print("-" * 70)

    print(
        f"Training matrix   : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Validation matrix : "
        f"{X_valid_processed.shape}"
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

    except Exception:

        feature_names = [
            f"feature_{i}"
            for i in range(
                X_train_processed.shape[1]
            )
        ]

    # --------------------------------------------------------
    # Create processed DataFrames
    # --------------------------------------------------------

    train_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
    )

    valid_df = pd.DataFrame(
        X_valid_processed,
        columns=feature_names,
    )

    # --------------------------------------------------------
    # Add target
    # --------------------------------------------------------

    train_df[TARGET_COLUMN] = (
        y_train.to_numpy()
    )

    valid_df[TARGET_COLUMN] = (
        y_valid.to_numpy()
    )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    train_output = (
        OUTPUT_DIR
        / "fraud_train_processed.csv"
    )

    valid_output = (
        OUTPUT_DIR
        / "fraud_validation_processed.csv"
    )

    pipeline_file = (
        OUTPUT_DIR
        / "fraud_preprocessor.joblib"
    )

    metadata_file = (
        OUTPUT_DIR
        / "fraud_preprocessing_metadata.json"
    )

    # --------------------------------------------------------
    # Save processed datasets
    # --------------------------------------------------------

    print("\nSaving processed datasets...")

    train_df.to_csv(
        train_output,
        index=False,
    )

    valid_df.to_csv(
        valid_output,
        index=False,
    )

    # --------------------------------------------------------
    # Save preprocessor
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        pipeline_file,
    )

    # --------------------------------------------------------
    # Target distribution metadata
    # --------------------------------------------------------

    target_distribution = {
        "train": {
            str(key): int(value)
            for key, value
            in (
                y_train
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "validation": {
            str(key): int(value)
            for key, value
            in (
                y_valid
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "input_file": str(INPUT_FILE),

        "target_column": TARGET_COLUMN,

        "random_state": RANDOM_STATE,

        "test_size": TEST_SIZE,

        "input_rows": int(len(df)),

        "input_columns": int(len(df.columns)),

        "training_rows": int(len(X_train)),

        "validation_rows": int(len(X_valid)),

        "numeric_feature_count": int(
            len(numeric_columns)
        ),

        "categorical_feature_count": int(
            len(categorical_columns)
        ),

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "processed_training_shape": [
            int(value)
            for value
            in X_train_processed.shape
        ],

        "processed_validation_shape": [
            int(value)
            for value
            in X_valid_processed.shape
        ],

        "processed_feature_count": int(
            X_train_processed.shape[1]
        ),

        "feature_names": feature_names,

        "target_distribution": target_distribution,

        "preprocessing": {
            "numeric_imputation": "median",

            "numeric_scaling": "StandardScaler",

            "categorical_imputation": (
                "most_frequent"
            ),

            "categorical_encoding": (
                "OneHotEncoder"
            ),

            "unknown_categories": "ignore",

            "fit_preprocessor_on": (
                "training_data_only"
            ),
        },

        "outputs": {
            "training_dataset": str(
                train_output
            ),

            "validation_dataset": str(
                valid_output
            ),

            "preprocessor": str(
                pipeline_file
            ),

            "metadata": str(
                metadata_file
            ),
        },
    }

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with metadata_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\nVerifying outputs...")

    if not train_output.exists():
        raise RuntimeError(
            "Training output was not created."
        )

    if not valid_output.exists():
        raise RuntimeError(
            "Validation output was not created."
        )

    if not pipeline_file.exists():
        raise RuntimeError(
            "Preprocessor was not created."
        )

    if not metadata_file.exists():
        raise RuntimeError(
            "Metadata file was not created."
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FRAUD PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nTraining output   : "
        f"{train_output}"
    )

    print(
        f"Validation output : "
        f"{valid_output}"
    )

    print(
        f"Preprocessor      : "
        f"{pipeline_file}"
    )

    print(
        f"Metadata          : "
        f"{metadata_file}"
    )

    print(
        f"\nFinal feature count: "
        f"{X_train_processed.shape[1]}"
    )

    print(
        "\nFraud ML preprocessing "
        "pipeline completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()