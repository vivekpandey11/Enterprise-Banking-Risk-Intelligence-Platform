from pathlib import Path
import json

import joblib
import pandas as pd


# ============================================================
# AML MODEL EXPLAINABILITY PIPELINE
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

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_demo"
    / "aml_demo_prediction.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_explainability"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "aml_feature_importance.csv"
)

TRANSACTION_EXPLANATION_FILE = (
    OUTPUT_DIR
    / "aml_transaction_explanation.json"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "aml_explainability_metadata.json"
)


def main():

    print("# AML MODEL EXPLAINABILITY PIPELINE")
    print()

    print(f"Project root: {PROJECT_ROOT}")
    print()

    # --------------------------------------------------------
    # Check required artifacts
    # --------------------------------------------------------

    print("Checking required artifacts...")

    required_files = [
        MODEL_FILE,
        PREPROCESSOR_FILE,
        PREDICTION_FILE,
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

    model = joblib.load(MODEL_FILE)

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
    # Load prediction
    # --------------------------------------------------------

    print("Loading AML demo prediction...")

    with open(
        PREDICTION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        prediction = json.load(f)

    probability = float(
        prediction.get(
            "probability",
            0.0
        )
    )

    risk_segment = prediction.get(
        "risk_segment",
        "UNKNOWN"
    )

    decision = prediction.get(
        "decision",
        "UNKNOWN"
    )

    print(
        f"AML probability : "
        f"{probability:.6f}"
    )

    print(
        f"Risk segment    : {risk_segment}"
    )

    print(
        f"Decision        : {decision}"
    )

    print()

    # --------------------------------------------------------
    # Extract feature names
    # --------------------------------------------------------

    print("Extracting feature names...")

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    print(
        f"Features explained: "
        f"{len(feature_names)}"
    )

    print()

    # --------------------------------------------------------
    # Global feature importance
    # --------------------------------------------------------

    print(
        "Extracting global feature importance..."
    )

    if hasattr(model, "coef_"):

        coefficients = model.coef_[0]

        importance_values = abs(
            coefficients
        )

        importance_type = (
            "absolute_logistic_regression_coefficient"
        )

    elif hasattr(model, "feature_importances_"):

        importance_values = (
            model.feature_importances_
        )

        importance_type = (
            "random_forest_feature_importance"
        )

    else:

        raise ValueError(
            "Model does not provide feature importance."
        )

    if len(feature_names) != len(
        importance_values
    ):

        raise ValueError(
            "Feature names and importance values "
            "are not aligned."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # Normalize importance for easier interpretation

    total_importance = (
        importance_df["importance"].sum()
    )

    if total_importance > 0:

        importance_df[
            "importance_normalized"
        ] = (
            importance_df["importance"]
            / total_importance
        )

    else:

        importance_df[
            "importance_normalized"
        ] = 0.0

    # --------------------------------------------------------
    # Print top features
    # --------------------------------------------------------

    print()
    print("## Top AML Model Features")
    print()

    top_features = importance_df.head(10)

    for index, row in top_features.iterrows():

        print(
            f"{index + 1}. "
            f"{row['feature']:<50} "
            f"importance="
            f"{row['importance']:.6f}"
        )

    # --------------------------------------------------------
    # Business explanation
    # --------------------------------------------------------

    print()
    print("## Business Explanation")
    print()

    transaction = prediction.get(
        "transaction",
        {}
    )

    amount_received = float(
        transaction.get(
            "amount_received",
            0.0
        )
    )

    amount_paid = float(
        transaction.get(
            "amount_paid",
            0.0
        )
    )

    cross_border = int(
        transaction.get(
            "cross_border_transaction",
            0
        )
    )

    currency_conversion = int(
        transaction.get(
            "currency_conversion",
            0
        )
    )

    mismatch_flag = int(
        transaction.get(
            "amount_mismatch_flag",
            0
        )
    )

    high_value = int(
        transaction.get(
            "high_value_transaction",
            0
        )
    )

    if amount_received >= 100000:

        print(
            "- Transaction involves a "
            "high-value received amount."
        )

    else:

        print(
            "- Transaction received amount "
            "is not classified as extremely high."
        )

    if cross_border == 1:

        print(
            "- Transaction is cross-border."
        )

    else:

        print(
            "- Transaction is not cross-border."
        )

    if currency_conversion == 1:

        print(
            "- Currency conversion is involved."
        )

    else:

        print(
            "- No currency conversion is detected."
        )

    if mismatch_flag == 1:

        print(
            "- Difference between paid and received "
            "amounts is flagged."
        )

    else:

        print(
            "- No significant amount mismatch flag."
        )

    if high_value == 1:

        print(
            "- High-value transaction indicator is active."
        )

    if probability >= 0.50:

        print(
            "- AML probability exceeds the default "
            "alert threshold."
        )

    else:

        print(
            "- AML probability is below the default "
            "alert threshold."
        )

    print(
        f"- AML probability is "
        f"{probability * 100:.4f}%."
    )

    print(
        f"- Risk segment is classified as "
        f"{risk_segment}."
    )

    # --------------------------------------------------------
    # Transaction-level explanation
    # --------------------------------------------------------

    print()
    print(
        "Generating transaction-level explanation..."
    )

    explanation_reasons = []

    if cross_border == 1:

        explanation_reasons.append(
            "Cross-border transaction"
        )

    if currency_conversion == 1:

        explanation_reasons.append(
            "Currency conversion"
        )

    if mismatch_flag == 1:

        explanation_reasons.append(
            "Amount mismatch"
        )

    if high_value == 1:

        explanation_reasons.append(
            "High-value transaction"
        )

    if not explanation_reasons:

        explanation_reasons.append(
            "No major AML risk indicator detected"
        )

    # --------------------------------------------------------
    # Top model features
    # --------------------------------------------------------

    top_feature_records = []

    for _, row in importance_df.head(10).iterrows():

        top_feature_records.append(
            {
                "feature": row["feature"],
                "importance": float(
                    row["importance"]
                ),
                "importance_normalized": float(
                    row["importance_normalized"]
                ),
            }
        )

    # --------------------------------------------------------
    # Create explanation artifact
    # --------------------------------------------------------

    transaction_explanation = {

        "pipeline":
            "AML model explainability",

        "model":
            type(model).__name__,

        "aml_probability":
            probability,

        "aml_probability_percent":
            probability * 100,

        "risk_segment":
            risk_segment,

        "decision":
            decision,

        "transaction_amount_received":
            amount_received,

        "transaction_amount_paid":
            amount_paid,

        "cross_border_transaction":
            bool(cross_border),

        "currency_conversion":
            bool(currency_conversion),

        "amount_mismatch_flag":
            bool(mismatch_flag),

        "high_value_transaction":
            bool(high_value),

        "business_risk_indicators":
            explanation_reasons,

        "top_model_features":
            top_feature_records,

        "explanation_summary":
            (
                "The AML model classified the transaction "
                f"as {risk_segment}. The predicted AML "
                f"probability is {probability * 100:.4f}%. "
                "The result should be interpreted with "
                "caution because the available validation "
                "dataset contains only one positive AML case."
            ),
    }

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save feature importance
    # --------------------------------------------------------

    print()
    print("Saving feature importance...")

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save transaction explanation
    # --------------------------------------------------------

    print(
        "Saving transaction explanation..."
    )

    with open(
        TRANSACTION_EXPLANATION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            transaction_explanation,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "pipeline":
            "AML explainability",

        "model_type":
            type(model).__name__,

        "importance_type":
            importance_type,

        "feature_count":
            len(feature_names),

        "top_features":
            10,

        "prediction_file":
            str(PREDICTION_FILE),

        "feature_importance_file":
            str(FEATURE_IMPORTANCE_FILE),

        "transaction_explanation_file":
            str(TRANSACTION_EXPLANATION_FILE),

        "validation_warning":
            (
                "AML validation contains only one "
                "positive case. Model and threshold "
                "performance are not production-grade."
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

    output_files = [
        FEATURE_IMPORTANCE_FILE,
        TRANSACTION_EXPLANATION_FILE,
        METADATA_FILE,
    ]

    for file_path in output_files:

        if not file_path.exists():

            raise RuntimeError(
                f"Expected output not created: "
                f"{file_path}"
            )

    print("All output artifacts verified.")
    print()

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("## Output artifacts")
    print()

    print(
        f"Feature importance:"
        f"\n{FEATURE_IMPORTANCE_FILE}"
    )

    print()

    print(
        f"Transaction explanation:"
        f"\n{TRANSACTION_EXPLANATION_FILE}"
    )

    print()

    print(
        f"Metadata:"
        f"\n{METADATA_FILE}"
    )

    print()

    print(
        "AML model explainability pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()