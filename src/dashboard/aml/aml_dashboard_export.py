from pathlib import Path
import json

import pandas as pd


# ============================================================
# AML EXECUTIVE DASHBOARD EXPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STAGING_DIR = PROJECT_ROOT / "data" / "staging" / "aml"
DASHBOARD_DIR = PROJECT_ROOT / "dashboards" / "aml"


# ------------------------------------------------------------
# Input artifacts
# ------------------------------------------------------------

BUSINESS_REPORT = (
    STAGING_DIR
    / "aml_reporting"
    / "aml_business_report.json"
)

BUSINESS_SUMMARY = (
    STAGING_DIR
    / "aml_reporting"
    / "aml_business_summary.csv"
)

ALERT_SUMMARY = (
    STAGING_DIR
    / "aml_reporting"
    / "aml_alert_summary.csv"
)

FEATURE_IMPORTANCE = (
    STAGING_DIR
    / "aml_explainability"
    / "aml_feature_importance.csv"
)

THRESHOLD_METADATA = (
    STAGING_DIR
    / "aml_threshold"
    / "aml_threshold_metadata.json"
)

MODEL_METADATA = (
    STAGING_DIR
    / "aml_model_metadata.json"
)

PROFILE_FILE = (
    STAGING_DIR
    / "aml_transactions_profile.json"
)


# ------------------------------------------------------------
# Output artifacts
# ------------------------------------------------------------

DASHBOARD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DASHBOARD_JSON = (
    DASHBOARD_DIR
    / "aml_executive_dashboard.json"
)

DASHBOARD_SUMMARY = (
    DASHBOARD_DIR
    / "aml_executive_summary.csv"
)

DASHBOARD_ALERTS = (
    DASHBOARD_DIR
    / "aml_alert_summary.csv"
)

DASHBOARD_FEATURES = (
    DASHBOARD_DIR
    / "aml_feature_importance.csv"
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def load_json(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def safe_number(value, default=0):
    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return int(value)

        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return default


def get_dataset_rows(profile):
    """
    Supports:
    - final_rows
    - input_rows
    - dataset_rows
    - rows
    """

    for key in [
        "final_rows",
        "input_rows",
        "dataset_rows",
        "rows",
    ]:

        value = profile.get(key)

        if value is not None:

            try:
                return int(value)

            except (
                TypeError,
                ValueError
            ):
                pass

    return 0


def get_dataset_column_count(profile):
    """
    Supports:
    - final_columns as integer
    - input_columns as integer
    - dataset_columns as integer
    - columns as list
    """

    # Preferred metadata fields
    for key in [
        "final_columns",
        "input_columns",
        "dataset_columns",
    ]:

        value = profile.get(key)

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        if isinstance(value, list):
            return len(value)

        if isinstance(value, tuple):
            return len(value)

        if value is not None:

            try:
                return int(value)

            except (
                TypeError,
                ValueError
            ):
                pass

    # Fallback: actual column list
    columns = profile.get(
        "columns",
        []
    )

    if isinstance(columns, list):
        return len(columns)

    if isinstance(columns, tuple):
        return len(columns)

    return 0


def get_dataset_columns(profile):

    columns = profile.get(
        "columns",
        []
    )

    if isinstance(columns, list):
        return columns

    if isinstance(columns, tuple):
        return list(columns)

    return []


# ============================================================
# MAIN
# ============================================================

def main():

    print("# AML EXECUTIVE DASHBOARD EXPORT")
    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print()

    # --------------------------------------------------------
    # Check artifacts
    # --------------------------------------------------------

    print(
        "Checking required dashboard artifacts..."
    )
    print()

    required_artifacts = {

        "aml_business_report.json":
            BUSINESS_REPORT,

        "aml_business_summary.csv":
            BUSINESS_SUMMARY,

        "aml_alert_summary.csv":
            ALERT_SUMMARY,

        "aml_feature_importance.csv":
            FEATURE_IMPORTANCE,

        "aml_threshold_metadata.json":
            THRESHOLD_METADATA,

        "aml_model_metadata.json":
            MODEL_METADATA,

        "aml_transactions_profile.json":
            PROFILE_FILE,
    }

    for name, path in required_artifacts.items():

        if path.exists():

            print(
                f"[FOUND] {name}"
            )

        else:

            raise FileNotFoundError(
                f"Required dashboard artifact not found:\n"
                f"{path}"
            )

    print()
    print(
        "All required artifacts found."
    )
    print()

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    print(
        "Loading business report..."
    )

    business_report = load_json(
        BUSINESS_REPORT
    )

    print(
        "Loading business summary..."
    )

    business_summary_df = pd.read_csv(
        BUSINESS_SUMMARY
    )

    print(
        "Loading alert summary..."
    )

    alert_summary_df = pd.read_csv(
        ALERT_SUMMARY
    )

    print(
        "Loading feature importance..."
    )

    feature_importance_df = pd.read_csv(
        FEATURE_IMPORTANCE
    )

    print(
        "Loading threshold metadata..."
    )

    threshold_metadata = load_json(
        THRESHOLD_METADATA
    )

    print(
        "Loading model metadata..."
    )

    model_metadata = load_json(
        MODEL_METADATA
    )

    print(
        "Loading AML profile..."
    )

    aml_profile = load_json(
        PROFILE_FILE
    )

    print()

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    dataset_rows = get_dataset_rows(
        aml_profile
    )

    dataset_columns = get_dataset_column_count(
        aml_profile
    )

    dataset_column_names = get_dataset_columns(
        aml_profile
    )

    # --------------------------------------------------------
    # Target information
    # --------------------------------------------------------

    legitimate_transactions = int(
        aml_profile.get(
            "legitimate_transactions",
            0
        )
    )

    laundering_transactions = int(
        aml_profile.get(
            "laundering_transactions",
            0
        )
    )

    # --------------------------------------------------------
    # Laundering rate
    # --------------------------------------------------------

    if dataset_rows > 0:

        laundering_rate = (
            laundering_transactions
            / dataset_rows
        ) * 100

    else:

        laundering_rate = 0.0

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    model_type = model_metadata.get(
        "best_model",
        model_metadata.get(
            "model_type",
            "Unknown"
        )
    )

    # --------------------------------------------------------
    # Threshold information
    # --------------------------------------------------------

    default_threshold = safe_number(
        threshold_metadata.get(
            "default_threshold",
            0.50
        ),
        0.50
    )

    best_f1_threshold = safe_number(
        threshold_metadata.get(
            "best_f1_threshold",
            default_threshold
        ),
        default_threshold
    )

    business_threshold = safe_number(
        threshold_metadata.get(
            "business_threshold",
            best_f1_threshold
        ),
        best_f1_threshold
    )

    # --------------------------------------------------------
    # Business risk information
    # --------------------------------------------------------

    aml_probability = safe_number(
        business_report.get(
            "aml_probability",
            business_report.get(
                "AML Probability",
                0
            )
        )
    )

    risk_segment = business_report.get(
        "risk_segment",
        business_report.get(
            "Risk Segment",
            "LOW RISK"
        )
    )

    business_risk = business_report.get(
        "business_risk",
        business_report.get(
            "Business Risk",
            "LOW_RISK"
        )
    )

    aml_decision = business_report.get(
        "aml_decision",
        business_report.get(
            "AML Decision",
            "LOW_RISK"
        )
    )

    alert_status = business_report.get(
        "alert_status",
        business_report.get(
            "Alert Status",
            "NO_ALERT"
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    top_features = []

    if not feature_importance_df.empty:

        for _, row in feature_importance_df.head(
            10
        ).iterrows():

            feature_name = str(
                row.iloc[0]
            )

            importance = safe_number(
                row.iloc[1]
            )

            top_features.append(
                {
                    "feature": feature_name,
                    "importance": importance,
                }
            )

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    dataset_statistics = {

        "rows": dataset_rows,

        "columns": dataset_columns,

        "column_names": dataset_column_names,

        "legitimate_transactions":
            legitimate_transactions,

        "laundering_transactions":
            laundering_transactions,

        "laundering_rate_percent":
            laundering_rate,
    }

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    model_summary = {

        "model_type":
            model_type,

        "default_threshold":
            default_threshold,

        "best_f1_threshold":
            best_f1_threshold,

        "business_threshold":
            business_threshold,
    }

    # --------------------------------------------------------
    # Business summary
    # --------------------------------------------------------

    business_summary = {

        "aml_probability":
            aml_probability,

        "risk_segment":
            risk_segment,

        "business_risk":
            business_risk,

        "aml_decision":
            aml_decision,

        "alert_status":
            alert_status,

        "business_threshold":
            business_threshold,
    }

    # --------------------------------------------------------
    # Executive dashboard payload
    # --------------------------------------------------------

    dashboard = {

        "dashboard": {

            "name":
                "AML Executive Risk Dashboard",

            "version":
                "1.0",

            "project":
                "Enterprise Banking Risk "
                "Intelligence Platform",
        },

        "dataset":
            dataset_statistics,

        "model":
            model_summary,

        "business_risk":
            business_summary,

        "top_features":
            top_features,

        "business_summary":
            business_summary_df.to_dict(
                orient="records"
            ),

        "alert_summary":
            alert_summary_df.to_dict(
                orient="records"
            ),

        "feature_importance":
            feature_importance_df.to_dict(
                orient="records"
            ),
    }

    # --------------------------------------------------------
    # Print executive summary
    # --------------------------------------------------------

    print(
        "## AML EXECUTIVE DASHBOARD SUMMARY"
    )
    print()

    print(
        f"Dataset rows       : "
        f"{dataset_rows:,}"
    )

    print(
        f"Dataset columns    : "
        f"{dataset_columns}"
    )

    print(
        f"Legitimate         : "
        f"{legitimate_transactions:,}"
    )

    print(
        f"Laundering         : "
        f"{laundering_transactions:,}"
    )

    print(
        f"Laundering rate    : "
        f"{laundering_rate:.6f}%"
    )

    print(
        f"Model              : "
        f"{model_type}"
    )

    print(
        f"AML Probability    : "
        f"{aml_probability:.6f}%"
    )

    print(
        f"Risk Segment       : "
        f"{risk_segment}"
    )

    print(
        f"AML Decision       : "
        f"{aml_decision}"
    )

    print(
        f"Alert Status       : "
        f"{alert_status}"
    )

    print(
        f"Business Threshold : "
        f"{business_threshold:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Save dashboard JSON
    # --------------------------------------------------------

    print(
        "Saving dashboard export..."
    )

    with open(
        DASHBOARD_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dashboard,
            f,
            indent=2,
            default=str
        )

    # --------------------------------------------------------
    # Save dashboard summary CSV
    # --------------------------------------------------------

    dashboard_summary_df = pd.DataFrame(
        [
            {

                "dataset_rows":
                    dataset_rows,

                "dataset_columns":
                    dataset_columns,

                "legitimate_transactions":
                    legitimate_transactions,

                "laundering_transactions":
                    laundering_transactions,

                "laundering_rate_percent":
                    laundering_rate,

                "model_type":
                    model_type,

                "aml_probability":
                    aml_probability,

                "risk_segment":
                    risk_segment,

                "business_risk":
                    business_risk,

                "aml_decision":
                    aml_decision,

                "alert_status":
                    alert_status,

                "business_threshold":
                    business_threshold,
            }
        ]
    )

    dashboard_summary_df.to_csv(
        DASHBOARD_SUMMARY,
        index=False
    )

    # --------------------------------------------------------
    # Save alerts
    # --------------------------------------------------------

    alert_summary_df.to_csv(
        DASHBOARD_ALERTS,
        index=False
    )

    # --------------------------------------------------------
    # Save feature importance
    # --------------------------------------------------------

    feature_importance_df.to_csv(
        DASHBOARD_FEATURES,
        index=False
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print()
    print(
        "Verifying outputs..."
    )

    output_files = [

        DASHBOARD_JSON,

        DASHBOARD_SUMMARY,

        DASHBOARD_ALERTS,

        DASHBOARD_FEATURES,
    ]

    for file_path in output_files:

        if not file_path.exists():

            raise RuntimeError(
                f"Expected output not created:\n"
                f"{file_path}"
            )

        if file_path.stat().st_size == 0:

            raise RuntimeError(
                f"Output file is empty:\n"
                f"{file_path}"
            )

    print(
        "All output artifacts verified."
    )
    print()

    print(
        "## Dashboard output artifacts"
    )
    print()

    print(
        f"Dashboard JSON : "
        f"{DASHBOARD_JSON}"
    )

    print(
        f"Summary CSV    : "
        f"{DASHBOARD_SUMMARY}"
    )

    print(
        f"Alert CSV      : "
        f"{DASHBOARD_ALERTS}"
    )

    print(
        f"Features CSV   : "
        f"{DASHBOARD_FEATURES}"
    )

    print()

    print(
        "AML executive dashboard export "
        "completed successfully."
    )


if __name__ == "__main__":
    main()