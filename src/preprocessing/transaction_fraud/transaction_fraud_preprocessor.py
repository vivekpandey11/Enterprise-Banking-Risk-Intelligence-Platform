from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# TRANSACTION FRAUD PREPROCESSING PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transaction_fraud"
    / "creditcard.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transaction_fraud"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_FILE = (
    OUTPUT_DIR
    / "transaction_fraud_train_processed.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "transaction_fraud_validation_processed.csv"
)

PREPROCESSOR_FILE = (
    OUTPUT_DIR
    / "transaction_fraud_preprocessor.joblib"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "transaction_fraud_preprocessing_metadata.json"
)

TARGET_COLUMN = "Class"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


# ============================================================
# VALIDATION
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found:\n{path}"
        )


def validate_dataset(df):

    required_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    invalid_targets = set(
        df[TARGET_COLUMN].unique()
    ) - {0, 1}

    if invalid_targets:
        raise ValueError(
            f"Unexpected target values: {invalid_targets}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TRANSACTION FRAUD PREPROCESSING PIPELINE")
    print("=" * 75)

    require_file(RAW_FILE)

    print("\nLoading raw fraud dataset...")
    print(RAW_FILE)

    df = pd.read_csv(RAW_FILE)

    print(
        f"\nRows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Validate schema
    # --------------------------------------------------------

    print("\nValidating dataset...")

    validate_dataset(df)

    print("Dataset schema: VALID")

    # --------------------------------------------------------
    # Remove duplicate transactions
    # --------------------------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    print(
        f"\nDuplicate rows found: {duplicate_count:,}"
    )

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(
            drop=True
        )

    print(
        f"Rows after duplicate removal: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Separate target
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN].astype(int)

    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    numeric_features = list(
        X.columns
    )

    # --------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------

    print("\nCreating stratified train/validation split...")

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        f"Training rows   : {len(X_train):,}"
    )

    print(
        f"Validation rows : {len(X_valid):,}"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution")

    print("\nFull dataset:")
    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

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
    # Preprocessor
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        ],
        remainder="drop",
    )

    # --------------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    print(
        "\nFitting preprocessor on training data only..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_valid_processed = (
        preprocessor.transform(
            X_valid
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_valid_processed = pd.DataFrame(
        X_valid_processed,
        columns=feature_names,
        index=X_valid.index,
    )

    train_processed = pd.concat(
        [
            X_train_processed,
            y_train.rename(TARGET_COLUMN),
        ],
        axis=1,
    )

    validation_processed = pd.concat(
        [
            X_valid_processed,
            y_valid.rename(TARGET_COLUMN),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    print("\nSaving processed datasets...")

    train_processed.to_csv(
        TRAIN_FILE,
        index=False,
    )

    validation_processed.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "pipeline": (
            "Transaction Fraud Preprocessing Pipeline"
        ),
        "input_file": str(RAW_FILE),
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "duplicate_rows_removed": duplicate_count,
        "target_distribution": {
            "full": {
                str(k): int(v)
                for k, v in y.value_counts()
                .sort_index()
                .items()
            },
            "train": {
                str(k): int(v)
                for k, v in y_train.value_counts()
                .sort_index()
                .items()
            },
            "validation": {
                str(k): int(v)
                for k, v in y_valid.value_counts()
                .sort_index()
                .items()
            },
        },
        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
            "fit_preprocessor_on": (
                "training_data_only"
            ),
            "split_strategy": "stratified",
        },
        "outputs": {
            "training_dataset": str(
                TRAIN_FILE
            ),
            "validation_dataset": str(
                VALIDATION_FILE
            ),
            "preprocessor": str(
                PREPROCESSOR_FILE
            ),
            "metadata": str(
                METADATA_FILE
            ),
        },
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TRANSACTION FRAUD PREPROCESSING COMPLETE")
    print("=" * 75)

    print(
        f"\nProcessed features : "
        f"{len(feature_names)}"
    )

    print(
        f"Training dataset   : "
        f"{TRAIN_FILE}"
    )

    print(
        f"Validation dataset : "
        f"{VALIDATION_FILE}"
    )

    print(
        f"Preprocessor       : "
        f"{PREPROCESSOR_FILE}"
    )

    print(
        f"Metadata            : "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()