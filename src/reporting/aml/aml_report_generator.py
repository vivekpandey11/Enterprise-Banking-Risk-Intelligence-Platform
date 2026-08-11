from pathlib import Path
import json
from datetime import datetime

import pandas as pd


# ============================================================
# AML BUSINESS REPORT GENERATOR
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ------------------------------------------------------------
# Input artifacts
# ------------------------------------------------------------

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_demo"
    / "aml_demo_prediction.json"
)

EXPLANATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_explainability"
    / "aml_transaction_explanation.json"
)

FEATURE_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_explainability"
    / "aml_feature_importance.csv"
)

THRESHOLD_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_threshold"
    / "aml_threshold_metadata.json"
)

MODEL_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_model_metadata.json"
)

# ------------------------------------------------------------
# Output artifacts
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
    / "aml_reporting"
)

REPORT_JSON = OUTPUT_DIR / "aml_business_report.json"
REPORT_CSV = OUTPUT_DIR / "aml_business_summary.csv"
ALERT_CSV = OUTPUT_DIR / "aml_alert_summary.csv"
METRICS_JSON = OUTPUT_DIR / "aml_model_metrics.json"


# ============================================================
# Utility functions
# ============================================================

def load_json(file_path: Path):
    """Load JSON artifact."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value, default=0.0):
    """Safely convert value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_value(data, *keys, default=None):
    """Get first available key from dictionary."""

    for key in keys:
        if isinstance(data, dict) and key in data:
            return data[key]

    return default


# ============================================================
# Main
# ============================================================

def main():

    print("# AML BUSINESS REPORT GENERATOR")
    print()

    print(f"Project root: {PROJECT_ROOT}")
    print()

    # --------------------------------------------------------
    # Check artifacts
    # --------------------------------------------------------

    print("Checking required artifacts...")
    print()

    required_files = [
        PREDICTION_FILE,
        EXPLANATION_FILE,
        FEATURE_IMPORTANCE_FILE,
        THRESHOLD_METADATA_FILE,
        MODEL_METADATA_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required artifact not found:\n{file_path}"
            )

        print(f"[FOUND] {file_path.name}")

    print()
    print("All required artifacts found.")
    print()

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    print("Loading AML prediction...")

    prediction = load_json(PREDICTION_FILE)

    print("Loading transaction explanation...")

    explanation = load_json(EXPLANATION_FILE)

    print("Loading threshold metadata...")

    threshold_metadata = load_json(
        THRESHOLD_METADATA_FILE
    )

    print("Loading model metadata...")

    model_metadata = load_json(
        MODEL_METADATA_FILE
    )

    print("Loading feature importance...")

    feature_importance = pd.read_csv(
        FEATURE_IMPORTANCE_FILE
    )

    print()

    # --------------------------------------------------------
    # Extract prediction information
    # --------------------------------------------------------

    probability = safe_float(
        get_value(
            prediction,
            "aml_probability",
            "fraud_probability",
            "probability",
            default=0.0,
        )
    )

    risk_segment = str(
        get_value(
            prediction,
            "risk_segment",
            "risk",
            default="UNKNOWN",
        )
    ).upper()

    decision = str(
        get_value(
            prediction,
            "aml_decision",
            "decision",
            default="UNKNOWN",
        )
    ).upper()

    aml_prediction = int(
        get_value(
            prediction,
            "aml_prediction",
            "prediction",
            default=0,
        )
    )

    # --------------------------------------------------------
    # Extract thresholds
    # --------------------------------------------------------

    default_threshold = safe_float(
        get_value(
            threshold_metadata,
            "default_threshold",
            default=0.50,
        ),
        0.50,
    )

    best_f1_threshold = safe_float(
        get_value(
            threshold_metadata,
            "best_f1_threshold",
            default=0.01,
        ),
        0.01,
    )

    business_threshold = safe_float(
        get_value(
            threshold_metadata,
            "business_threshold",
            "experimental_business_threshold",
            default=0.01,
        ),
        0.01,
    )

    # --------------------------------------------------------
    # Extract explanation information
    # --------------------------------------------------------

    business_explanation = get_value(
        explanation,
        "business_explanation",
        "explanation",
        default=[],
    )

    if isinstance(business_explanation, str):
        business_explanation = [
            business_explanation
        ]

    if not isinstance(business_explanation, list):
        business_explanation = []

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    model_type = get_value(
        model_metadata,
        "best_model",
        "model_type",
        default="LogisticRegression",
    )

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    if probability >= business_threshold:

        alert_status = "AML_ALERT"

    else:

        alert_status = "NO_ALERT"

    if probability >= 0.80:

        business_risk = "HIGH_RISK"

    elif probability >= 0.30:

        business_risk = "MEDIUM_RISK"

    else:

        business_risk = "LOW_RISK"

    # --------------------------------------------------------
    # Top features
    # --------------------------------------------------------

    top_features = []

    if not feature_importance.empty:

        importance_column = None

        for column in [
            "importance",
            "coefficient",
            "feature_importance",
        ]:

            if column in feature_importance.columns:
                importance_column = column
                break

        feature_column = None

        for column in [
            "feature",
            "feature_name",
            "name",
        ]:

            if column in feature_importance.columns:
                feature_column = column
                break

        if importance_column and feature_column:

            temp = feature_importance.copy()

            temp[importance_column] = pd.to_numeric(
                temp[importance_column],
                errors="coerce",
            )

            temp = temp.dropna(
                subset=[importance_column]
            )

            temp = temp.sort_values(
                importance_column,
                ascending=False,
            ).head(10)

            for _, row in temp.iterrows():

                top_features.append(
                    {
                        "feature": str(
                            row[feature_column]
                        ),
                        "importance": round(
                            float(
                                row[importance_column]
                            ),
                            6,
                        ),
                    }
                )

    # --------------------------------------------------------
    # Build business report
    # --------------------------------------------------------

    report = {

        "report_type": "AML Business Risk Report",

        "generated_at": datetime.now().isoformat(),

        "project": (
            "Enterprise Banking Risk "
            "Intelligence Platform"
        ),

        "model": {
            "model_type": str(model_type),
            "default_threshold": default_threshold,
            "best_f1_threshold": best_f1_threshold,
            "business_threshold": business_threshold,
        },

        "transaction": {

            "aml_probability": round(
                probability,
                8,
            ),

            "aml_probability_percent": round(
                probability * 100,
                6,
            ),

            "model_prediction": aml_prediction,

            "model_decision": decision,

            "risk_segment": risk_segment,

            "business_risk": business_risk,

            "alert_status": alert_status,
        },

        "business_explanation": business_explanation,

        "top_model_features": top_features,

        "governance_note": (
            "Current AML validation performance is "
            "not production-grade because the available "
            "sample contains only a very small number of "
            "positive laundering transactions. "
            "Threshold results should therefore be treated "
            "as experimental until evaluated on a larger "
            "and more representative AML dataset."
        ),
    }

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("## AML BUSINESS RISK SUMMARY")
    print()

    print(
        f"AML Probability : "
        f"{probability * 100:.6f}%"
    )

    print(
        f"Risk Segment    : "
        f"{risk_segment}"
    )

    print(
        f"Business Risk   : "
        f"{business_risk}"
    )

    print(
        f"AML Decision    : "
        f"{decision}"
    )

    print(
        f"Alert Status    : "
        f"{alert_status}"
    )

    print(
        f"Business Threshold : "
        f"{business_threshold:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------

    print("Saving business report...")

    with open(
        REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Business summary CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "generated_at": report["generated_at"],
                "model_type": model_type,
                "aml_probability": probability,
                "aml_probability_percent": (
                    probability * 100
                ),
                "risk_segment": risk_segment,
                "business_risk": business_risk,
                "aml_decision": decision,
                "alert_status": alert_status,
                "default_threshold": default_threshold,
                "best_f1_threshold": best_f1_threshold,
                "business_threshold": business_threshold,
            }
        ]
    )

    summary_df.to_csv(
        REPORT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Alert summary CSV
    # --------------------------------------------------------

    alert_df = pd.DataFrame(
        [
            {
                "alert_id": "AML-DEMO-001",
                "aml_probability": probability,
                "risk_segment": business_risk,
                "alert_status": alert_status,
                "aml_decision": decision,
                "requires_review": (
                    alert_status == "AML_ALERT"
                ),
            }
        ]
    )

    alert_df.to_csv(
        ALERT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    comparison_file = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "aml"
        / "aml_model_comparison.csv"
    )

    metrics = {
        "model_type": str(model_type),
        "default_threshold": default_threshold,
        "best_f1_threshold": best_f1_threshold,
        "business_threshold": business_threshold,
        "validation_warning": (
            "Validation contains only one positive AML "
            "case; metrics are unstable."
        ),
    }

    if comparison_file.exists():

        try:

            comparison_df = pd.read_csv(
                comparison_file
            )

            metrics[
                "model_comparison_rows"
            ] = int(len(comparison_df))

        except Exception:

            metrics[
                "model_comparison_rows"
            ] = None

    with open(
        METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    print()
    print("Verifying outputs...")

    output_files = [
        REPORT_JSON,
        REPORT_CSV,
        ALERT_CSV,
        METRICS_JSON,
    ]

    for file_path in output_files:

        if not file_path.exists():

            raise RuntimeError(
                f"Expected output not created:\n"
                f"{file_path}"
            )

    print("All output artifacts verified.")
    print()

    print("## Output artifacts")
    print()

    print(
        f"Business report : {REPORT_JSON}"
    )

    print(
        f"Business summary: {REPORT_CSV}"
    )

    print(
        f"Alert summary   : {ALERT_CSV}"
    )

    print(
        f"Model metrics   : {METRICS_JSON}"
    )

    print()

    print(
        "AML business reporting pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()