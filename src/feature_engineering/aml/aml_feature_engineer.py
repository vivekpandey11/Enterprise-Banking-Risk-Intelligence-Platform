from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_transactions_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "aml"

OUTPUT_FILE = OUTPUT_DIR / "aml_features.csv"
METADATA_FILE = OUTPUT_DIR / "aml_feature_metadata.json"


def main():

    print("# AML FEATURE ENGINEERING PIPELINE")
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading clean AML dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows    : {len(df):,}")
    print(f"Input columns : {len(df.columns)}")
    print()

    original_columns = df.columns.tolist()

    # ---------------------------------------------------------
    # Timestamp features
    # ---------------------------------------------------------

    print("Creating timestamp features...")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["transaction_hour"] = (
        df["timestamp"].dt.hour
    )

    df["transaction_day"] = (
        df["timestamp"].dt.day
    )

    df["transaction_month"] = (
        df["timestamp"].dt.month
    )

    df["transaction_day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["transaction_day_of_week"] >= 5
    ).astype(int)

    # ---------------------------------------------------------
    # Amount features
    # ---------------------------------------------------------

    print("Creating transaction amount features...")

    df["amount_difference"] = (
        df["amount_paid"] -
        df["amount_received"]
    )

    df["amount_difference_abs"] = (
        df["amount_difference"].abs()
    )

    df["amount_ratio"] = (
        df["amount_received"] /
        df["amount_paid"].replace(0, np.nan)
    )

    df["amount_ratio"] = (
        df["amount_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["log_amount_received"] = np.log1p(
        df["amount_received"].clip(lower=0)
    )

    df["log_amount_paid"] = np.log1p(
        df["amount_paid"].clip(lower=0)
    )

    # ---------------------------------------------------------
    # Country relationship features
    # ---------------------------------------------------------

    print("Creating country relationship features...")

    df["cross_border_transaction"] = (
        df["from_country"] !=
        df["to_country"]
    ).astype(int)

    df["same_country"] = (
        df["from_country"] ==
        df["to_country"]
    ).astype(int)

    # ---------------------------------------------------------
    # Currency relationship features
    # ---------------------------------------------------------

    print("Creating currency relationship features...")

    df["currency_conversion"] = (
        df["payment_currency"] !=
        df["receiving_currency"]
    ).astype(int)

    df["same_currency"] = (
        df["payment_currency"] ==
        df["receiving_currency"]
    ).astype(int)

    # ---------------------------------------------------------
    # Bank relationship features
    # ---------------------------------------------------------

    print("Creating bank relationship features...")

    df["same_bank"] = (
        df["from_bank"] ==
        df["to_bank"]
    ).astype(int)

    # ---------------------------------------------------------
    # Account relationship features
    # ---------------------------------------------------------

    print("Creating account relationship features...")

    df["same_account"] = (
        df["from_account"] ==
        df["to_account"]
    ).astype(int)

    # ---------------------------------------------------------
    # Transaction risk indicators
    # ---------------------------------------------------------

    print("Creating AML risk indicators...")

    df["high_value_transaction"] = (
        df["amount_received"] >
        df["amount_received"].quantile(0.95)
    ).astype(int)

    df["amount_mismatch_flag"] = (
        df["amount_difference_abs"] >
        df["amount_received"] * 0.10
    ).astype(int)

    df["cross_border_currency_flag"] = (
        (
            df["cross_border_transaction"] == 1
        )
        &
        (
            df["currency_conversion"] == 1
        )
    ).astype(int)

    # ---------------------------------------------------------
    # Payment format
    # ---------------------------------------------------------

    print("Encoding payment format...")

    payment_format_dummies = pd.get_dummies(
        df["payment_format"],
        prefix="payment_format",
        dtype=int
    )

    df = pd.concat(
        [
            df,
            payment_format_dummies
        ],
        axis=1
    )

    # ---------------------------------------------------------
    # Preserve target
    # ---------------------------------------------------------

    df["is_laundering"] = (
        df["is_laundering"].astype(int)
    )

    # ---------------------------------------------------------
    # Remove leakage / non-feature columns
    # ---------------------------------------------------------

    print("Removing leakage-prone columns...")

    columns_to_drop = [
        "predicted_alert",
        "model_score",
        "is_dashboard_sample",
        "timestamp",
        "payment_format",
    ]

    columns_to_drop = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    df = df.drop(
        columns=columns_to_drop
    )

    # ---------------------------------------------------------
    # Convert boolean columns
    # ---------------------------------------------------------

    bool_columns = df.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    for column in bool_columns:
        df[column] = df[column].astype(int)

    # ---------------------------------------------------------
    # Handle missing / infinite values
    # ---------------------------------------------------------

    print("Checking engineered features...")

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    df[numeric_columns] = (
        df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if df.isna().sum().sum() != 0:
        raise ValueError(
            "Missing values remain after feature engineering."
        )

    if np.isinf(
        df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).sum() != 0:
        raise ValueError(
            "Infinite values remain after feature engineering."
        )

    # ---------------------------------------------------------
    # Target distribution
    # ---------------------------------------------------------

    print()
    print("Final target distribution:")

    print(
        df["is_laundering"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    # ---------------------------------------------------------
    # Feature list
    # ---------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column != "is_laundering"
    ]

    print(
        f"Final feature count: {len(feature_columns)}"
    )

    print()
    print("Engineered features:")

    for index, feature in enumerate(
        feature_columns,
        start=1
    ):
        print(
            f"{index:02d}. {feature}"
        )

    # ---------------------------------------------------------
    # Save dataset
    # ---------------------------------------------------------

    print()
    print("Saving engineered dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    metadata = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input_rows": int(len(pd.read_csv(INPUT_FILE))),
        "output_rows": int(len(df)),
        "original_columns": original_columns,
        "final_columns": df.columns.tolist(),
        "feature_columns": feature_columns,
        "target_column": "is_laundering",
        "target_distribution": {
            str(k): int(v)
            for k, v in df[
                "is_laundering"
            ].value_counts().items()
        },
        "removed_columns": columns_to_drop,
        "feature_engineering": [
            "timestamp features",
            "amount difference",
            "amount ratio",
            "log transaction amounts",
            "cross-border indicator",
            "currency conversion indicator",
            "same-bank indicator",
            "same-account indicator",
            "high-value transaction indicator",
            "amount mismatch indicator",
            "cross-border currency indicator",
            "payment-format one-hot encoding",
        ],
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print()
    print("Output artifacts:")
    print()
    print(
        f"Feature dataset : {OUTPUT_FILE}"
    )
    print(
        f"Metadata        : {METADATA_FILE}"
    )

    print()
    print(
        f"Final rows      : {len(df):,}"
    )

    print(
        f"Final columns   : {len(df.columns)}"
    )

    print(
        f"Final features  : {len(feature_columns)}"
    )

    print()
    print(
        "AML feature engineering pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()