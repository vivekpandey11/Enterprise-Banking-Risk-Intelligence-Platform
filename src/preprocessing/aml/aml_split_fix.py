from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml"
    / "aml_features.csv"
)

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml"
    / "aml_train"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml"
    / "aml_validation"
)

TRAIN_FILE = TRAIN_DIR / "aml_train_features.csv"
VALIDATION_FILE = VALIDATION_DIR / "aml_validation_features.csv"
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_split_metadata.json"
)


def main():

    print("# AML TRAIN / VALIDATION SPLIT FIX")
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print()

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"AML feature dataset not found:\n{INPUT_FILE}"
        )

    print("Loading AML feature dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows    : {len(df):,}")
    print(f"Input columns : {len(df.columns)}")
    print()

    # ---------------------------------------------------------
    # Validate target
    # ---------------------------------------------------------
    TARGET = "is_laundering"

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    print("Target distribution:")
    print(df[TARGET].value_counts(dropna=False).to_string())
    print()

    positive_count = int((df[TARGET] == 1).sum())
    negative_count = int((df[TARGET] == 0).sum())

    print(f"Positive AML rows : {positive_count:,}")
    print(f"Negative AML rows : {negative_count:,}")
    print()

    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------
    if positive_count < 2:
        raise ValueError(
            "Not enough positive AML transactions for a train/validation split."
        )

    # With only 3 positives, stratified 80/20 guarantees that
    # at least one positive row is placed into validation.
    test_size = 0.20

    print("Creating stratified train/validation split...")
    print(f"Validation size : {test_size:.0%}")
    print()

    train_df, validation_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[TARGET],
    )

    # ---------------------------------------------------------
    # Create directories
    # ---------------------------------------------------------
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Save datasets
    # ---------------------------------------------------------
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)

    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)

    # ---------------------------------------------------------
    # Distribution
    # ---------------------------------------------------------
    train_positive = int((train_df[TARGET] == 1).sum())
    train_negative = int((train_df[TARGET] == 0).sum())

    validation_positive = int((validation_df[TARGET] == 1).sum())
    validation_negative = int((validation_df[TARGET] == 0).sum())

    print("## Dataset split")
    print()
    print(f"Training rows   : {len(train_df):,}")
    print(f"Validation rows : {len(validation_df):,}")
    print()

    print("Training target distribution:")
    print(train_df[TARGET].value_counts().sort_index().to_string())
    print()

    print("Validation target distribution:")
    print(validation_df[TARGET].value_counts().sort_index().to_string())
    print()

    # ---------------------------------------------------------
    # Critical validation
    # ---------------------------------------------------------
    if validation_positive == 0:
        raise RuntimeError(
            "Validation set contains ZERO AML positive cases."
        )

    if train_positive == 0:
        raise RuntimeError(
            "Training set contains ZERO AML positive cases."
        )

    print("AML positive cases in validation: PASSED")
    print("AML positive cases in training   : PASSED")
    print()

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    metadata = {
        "input_file": str(INPUT_FILE),
        "train_file": str(TRAIN_FILE),
        "validation_file": str(VALIDATION_FILE),
        "random_state": 42,
        "validation_size": test_size,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_positive": positive_count,
        "total_negative": negative_count,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "train_positive": train_positive,
        "train_negative": train_negative,
        "validation_positive": validation_positive,
        "validation_negative": validation_negative,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------
    print("## Output artifacts")
    print()
    print(f"Training dataset   : {TRAIN_FILE}")
    print(f"Validation dataset : {VALIDATION_FILE}")
    print(f"Split metadata     : {METADATA_FILE}")
    print()

    print("AML train/validation split completed successfully.")


if __name__ == "__main__":
    main()