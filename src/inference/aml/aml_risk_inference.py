from pathlib import Path
import json

import joblib
import pandas as pd


# ============================================================
# AML RISK INFERENCE PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "aml"
    / "aml_logistic_regression.joblib"
)

PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml"
    / "aml_preprocessor.joblib"
)

THRESHOLD_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_threshold"
    / "aml_threshold_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_demo"
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "aml_demo_prediction.json"
)


def main():

    print("# AML RISK INFERENCE PIPELINE")
    print()

    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print()

    # --------------------------------------------------------
    # Check artifacts
    # --------------------------------------------------------

    print("Checking required artifacts...")

    required_files = [
        MODEL_FILE,
        PREPROCESSOR_FILE,
        THRESHOLD_METADATA_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required artifact not found:\n{file_path}"
            )

    print("All required artifacts found.")
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
    # Load preprocessor
    # --------------------------------------------------------

    print("Loading AML preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        f"Preprocessor type: "
        f"{type(preprocessor).__name__}"
    )
    print()

    # --------------------------------------------------------
    # Load threshold metadata
    # --------------------------------------------------------

    print("Loading threshold metadata...")

    with open(
        THRESHOLD_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        threshold_metadata = json.load(f)

    # IMPORTANT:
    # Because validation has only one AML positive,
    # the optimized threshold is experimental.
    #
    # We use 0.50 as the inference default instead of
    # blindly using the unstable 0.01 threshold.

    default_threshold = float(
        threshold_metadata.get(
            "default_threshold",
            0.50
        )
    )

    best_f1_threshold = float(
        threshold_metadata.get(
            "best_f1_threshold",
            0.50
        )
    )

    business_threshold = float(
        threshold_metadata.get(
            "business_threshold",
            0.50
        )
    )

    print(
        f"Default threshold    : {default_threshold:.2f}"
    )

    print(
        f"Best F1 threshold    : {best_f1_threshold:.2f}"
    )

    print(
        f"Experimental business: {business_threshold:.2f}"
    )

    print()

    # --------------------------------------------------------
    # Demo transaction
    # --------------------------------------------------------

    print("Creating demo AML transaction...")

    demo_transaction = {
        "from_country": "US",
        "to_country": "GB",
        "receiving_currency": "GBP",
        "payment_currency": "USD",

        "amount_received": 250000.00,
        "amount_paid": 300000.00,

        "transaction_hour": 2,
        "transaction_day": 15,
        "transaction_month": 8,
        "transaction_day_of_week": 5,

        "is_weekend": 0,

        "amount_difference": 50000.00,
        "amount_difference_abs": 50000.00,
        "amount_ratio": 1.20,

        "log_amount_received": 12.429,
        "log_amount_paid": 12.612,

        "cross_border_transaction": 1,
        "same_country": 0,

        "currency_conversion": 1,
        "same_currency": 0,

        "same_bank": 0,
        "same_account": 0,

        "high_value_transaction": 1,
        "amount_mismatch_flag": 1,
        "cross_border_currency_flag": 1,

        "payment_format_ACH": 0,
        "payment_format_Bitcoin": 0,
        "payment_format_Cash": 0,
        "payment_format_Cheque": 0,
        "payment_format_Credit Card": 0,
        "payment_format_Reinvestment": 0,
        "payment_format_Wire": 1,
    }

    demo_df = pd.DataFrame(
        [demo_transaction]
    )

    print(
        f"Input features: {len(demo_df.columns)}"
    )

    print()

    # --------------------------------------------------------
    # Align columns with preprocessor
    # --------------------------------------------------------

    try:

        expected_columns = (
            preprocessor.feature_names_in_
        )

        missing_columns = [
            col
            for col in expected_columns
            if col not in demo_df.columns
        ]

        extra_columns = [
            col
            for col in demo_df.columns
            if col not in expected_columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing input features:\n"
                + "\n".join(missing_columns)
            )

        if extra_columns:

            demo_df = demo_df.drop(
                columns=extra_columns
            )

        demo_df = demo_df[
            expected_columns
        ]

    except AttributeError:

        pass

    # --------------------------------------------------------
    # Transform transaction
    # --------------------------------------------------------

    print(
        "Transforming transaction input..."
    )

    X_transformed = preprocessor.transform(
        demo_df
    )

    print(
        f"Transformed feature count: "
        f"{X_transformed.shape[1]}"
    )

    print()

    # --------------------------------------------------------
    # Generate probability
    # --------------------------------------------------------

    print(
        "Generating AML probability..."
    )

    probability = float(
        model.predict_proba(
            X_transformed
        )[0][1]
    )

    print(
        f"AML Probability : {probability:.6f}"
    )

    print(
        f"AML Probability : {probability * 100:.4f}%"
    )

    print()

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    if probability >= default_threshold:

        risk_segment = "HIGH RISK"
        decision = "AML_ALERT"
        prediction = 1

    elif probability >= 0.20:

        risk_segment = "MEDIUM RISK"
        decision = "REVIEW"
        prediction = 0

    else:

        risk_segment = "LOW RISK"
        decision = "LOW_RISK"
        prediction = 0

    print(
        f"Risk Segment      : {risk_segment}"
    )

    print(
        f"Default Threshold : {default_threshold:.2f}"
    )

    print(
        f"AML Decision      : {decision}"
    )

    print(
        f"AML Prediction    : {prediction}"
    )

    print()

    # --------------------------------------------------------
    # Business interpretation
    # --------------------------------------------------------

    if prediction == 1:

        explanation = (
            "Transaction probability exceeds the "
            "default AML alert threshold."
        )

    elif risk_segment == "MEDIUM RISK":

        explanation = (
            "Transaction does not exceed the AML alert "
            "threshold but should receive additional review."
        )

    else:

        explanation = (
            "Transaction probability is below the "
            "AML alert threshold."
        )

    print(
        f"Business Explanation: {explanation}"
    )

    print()

    # --------------------------------------------------------
    # Prediction artifact
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_artifact = {

        "pipeline": "AML risk inference",

        "model": type(model).__name__,

        "model_file": str(
            MODEL_FILE
        ),

        "probability": probability,

        "probability_percent": (
            probability * 100
        ),

        "default_threshold": (
            default_threshold
        ),

        "experimental_best_f1_threshold": (
            best_f1_threshold
        ),

        "experimental_business_threshold": (
            business_threshold
        ),

        "risk_segment": risk_segment,

        "decision": decision,

        "prediction": prediction,

        "explanation": explanation,

        "validation_warning": (
            "Threshold optimization was performed "
            "on a validation set containing only one "
            "positive AML case. Thresholds are "
            "experimental and require a larger AML "
            "validation dataset before production use."
        ),

        "transaction": demo_transaction,
    }

    with open(
        PREDICTION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            prediction_artifact,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("Prediction artifact:")
    print(
        PREDICTION_FILE
    )

    print()

    print(
        "AML risk inference pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()