from pathlib import Path
import json

import pandas as pd

# ============================================================
# INTEGRATED EXECUTIVE RISK DASHBOARD
# Credit Risk + Fraud Risk + AML Risk
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INTEGRATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "integrated_risk"
)

CREDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
)

FRAUD_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "fraud"
)

AML_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "aml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dashboards"
    / "integrated"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

INTEGRATED_ASSESSMENT = (
    INTEGRATED_DIR
    / "integrated_risk_assessment.json"
)

INTEGRATED_SUMMARY = (
    INTEGRATED_DIR
    / "integrated_risk_summary.json"
)

# Optional business artifacts
AML_BUSINESS_SUMMARY = (
    AML_DIR
    / "aml_reporting"
    / "aml_business_summary.csv"
)

AML_ALERT_SUMMARY = (
    AML_DIR
    / "aml_reporting"
    / "aml_alert_summary.csv"
)

FRAUD_MODEL_EVALUATION = (
    FRAUD_DIR
    / "fraud_model_evaluation.json"
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

DASHBOARD_JSON = (
    OUTPUT_DIR
    / "integrated_executive_dashboard.json"
)

DASHBOARD_SUMMARY = (
    OUTPUT_DIR
    / "integrated_executive_summary.csv"
)

RISK_COMPONENTS = (
    OUTPUT_DIR
    / "integrated_risk_components.csv"
)

ALERT_SUMMARY = (
    OUTPUT_DIR
    / "integrated_alert_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (
        TypeError,
        ValueError
    ):
        return default


def find_nested(data, keys, default=None):
    """
    Search common locations for a value.
    """

    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data:
            return data[key]

    for section in [
        "integrated_risk",
        "credit_risk",
        "fraud_risk",
        "aml_risk",
        "business_risk",
        "model",
    ]:

        section_data = data.get(section)

        if isinstance(section_data, dict):

            for key in keys:

                if key in section_data:
                    return section_data[key]

    return default


def load_optional_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

def main():

    print("# INTEGRATED EXECUTIVE RISK DASHBOARD")
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print()

    # --------------------------------------------------------
    # Check artifacts
    # --------------------------------------------------------

    print("Checking required dashboard artifacts...")
    print()

    if not INTEGRATED_ASSESSMENT.exists():
        raise FileNotFoundError(
            f"Integrated assessment not found:\n"
            f"{INTEGRATED_ASSESSMENT}"
        )

    if not INTEGRATED_SUMMARY.exists():
        raise FileNotFoundError(
            f"Integrated summary not found:\n"
            f"{INTEGRATED_SUMMARY}"
        )

    print("[FOUND] integrated_risk_assessment.json")
    print("[FOUND] integrated_risk_summary.json")

    if AML_BUSINESS_SUMMARY.exists():
        print("[FOUND] AML business summary")

    if AML_ALERT_SUMMARY.exists():
        print("[FOUND] AML alert summary")

    print()
    print("All required artifacts found.")
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("Loading integrated risk assessment...")

    assessment = load_json(
        INTEGRATED_ASSESSMENT
    )

    print("Loading integrated risk summary...")

    summary = load_json(
        INTEGRATED_SUMMARY
    )

    print()

    # --------------------------------------------------------
    # Extract component risks
    # --------------------------------------------------------

    credit = assessment.get(
        "credit_risk",
        {}
    )

    fraud = assessment.get(
        "fraud_risk",
        {}
    )

    aml = assessment.get(
        "aml_risk",
        {}
    )

    integrated = assessment.get(
        "integrated_risk",
        {}
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    credit_score = safe_float(
        credit.get(
            "risk_score",
            summary.get(
                "credit_risk_score",
                0
            )
        )
    )

    fraud_score = safe_float(
        fraud.get(
            "risk_score",
            summary.get(
                "fraud_risk_score",
                0
            )
        )
    )

    aml_score = safe_float(
        aml.get(
            "risk_score",
            summary.get(
                "aml_risk_score",
                0
            )
        )
    )

    overall_score = safe_float(
        integrated.get(
            "overall_risk_score",
            summary.get(
                "overall_risk_score",
                0
            )
        )
    )

    risk_tier = integrated.get(
        "risk_tier",
        summary.get(
            "risk_tier",
            "UNKNOWN"
        )
    )

    alert_status = integrated.get(
        "alert_status",
        summary.get(
            "alert_status",
            "NO_ALERT"
        )
    )

    top_risk_driver = integrated.get(
        "top_risk_driver",
        summary.get(
            "top_risk_driver",
            "UNKNOWN"
        )
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    credit_probability = safe_float(
        credit.get(
            "probability",
            0
        )
    )

    fraud_probability = safe_float(
        fraud.get(
            "probability",
            0
        )
    )

    aml_probability = safe_float(
        aml.get(
            "probability",
            0
        )
    )

    # --------------------------------------------------------
    # Risk components table
    # --------------------------------------------------------

    components = pd.DataFrame(
        [
            {
                "risk_domain": "Credit Risk",
                "risk_score": credit_score,
                "probability": credit_probability,
                "risk_segment":
                    credit.get(
                        "risk_segment",
                        "UNKNOWN"
                    ),
                "decision":
                    credit.get(
                        "decision",
                        "UNKNOWN"
                    ),
                "weight": 0.35,
            },
            {
                "risk_domain": "Fraud Risk",
                "risk_score": fraud_score,
                "probability": fraud_probability,
                "risk_segment":
                    fraud.get(
                        "risk_segment",
                        "UNKNOWN"
                    ),
                "decision":
                    fraud.get(
                        "decision",
                        "UNKNOWN"
                    ),
                "weight": 0.35,
            },
            {
                "risk_domain": "AML Risk",
                "risk_score": aml_score,
                "probability": aml_probability,
                "risk_segment":
                    aml.get(
                        "risk_segment",
                        "UNKNOWN"
                    ),
                "decision":
                    aml.get(
                        "decision",
                        "UNKNOWN"
                    ),
                "weight": 0.30,
            },
        ]
    )

    # --------------------------------------------------------
    # Alert logic
    # --------------------------------------------------------

    alert_triggered = (
        alert_status != "NO_ALERT"
    )

    alert_df = pd.DataFrame(
        [
            {
                "overall_risk_score":
                    overall_score,
                "risk_tier":
                    risk_tier,
                "alert_status":
                    alert_status,
                "alert_triggered":
                    alert_triggered,
                "top_risk_driver":
                    top_risk_driver,
            }
        ]
    )

    # --------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------

    executive_summary = pd.DataFrame(
        [
            {
                "overall_risk_score":
                    overall_score,
                "risk_tier":
                    risk_tier,
                "alert_status":
                    alert_status,
                "top_risk_driver":
                    top_risk_driver,

                "credit_risk_score":
                    credit_score,

                "fraud_risk_score":
                    fraud_score,

                "aml_risk_score":
                    aml_score,

                "credit_probability":
                    credit_probability,

                "fraud_probability":
                    fraud_probability,

                "aml_probability":
                    aml_probability,
            }
        ]
    )

    # --------------------------------------------------------
    # Dashboard payload
    # --------------------------------------------------------

    dashboard = {

        "dashboard": {
            "name":
                "Integrated Executive Risk Dashboard",

            "version":
                "1.0",

            "project":
                "Enterprise Banking Risk "
                "Intelligence Platform",

            "purpose":
                "Unified executive view of "
                "credit, fraud and AML risk.",
        },

        "executive_summary": {
            "overall_risk_score":
                overall_score,

            "risk_tier":
                risk_tier,

            "alert_status":
                alert_status,

            "top_risk_driver":
                top_risk_driver,
        },

        "risk_domains": {
            "credit": {
                "score":
                    credit_score,
                "probability":
                    credit_probability,
                "risk_segment":
                    credit.get(
                        "risk_segment",
                        "UNKNOWN"
                    ),
                "decision":
                    credit.get(
                        "decision",
                        "UNKNOWN"
                    ),
                "weight": 0.35,
            },

            "fraud": {
                "score":
                    fraud_score,
                "probability":
                    fraud_probability,
                "risk_segment":
                    fraud.get(
                        "risk_segment",
                        "UNKNOWN"
                    ),
                "decision":
                    fraud.get(
                        "decision",
                        "UNKNOWN"
                    ),
                "weight": 0.35,
            },

            "aml": {
                "score":
                    aml_score,
                "probability":
                    aml_probability,
                "risk_segment":
                    aml.get(
                        "risk_segment",
                        "UNKNOWN"
                    ),
                "decision":
                    aml.get(
                        "decision",
                        "UNKNOWN"
                    ),
                "alert_status":
                    aml.get(
                        "alert_status",
                        "NO_ALERT"
                    ),
                "weight": 0.30,
            },
        },

        "risk_components":
            components.to_dict(
                orient="records"
            ),

        "alert_summary":
            alert_df.to_dict(
                orient="records"
            ),

        "source_assessment":
            assessment,

        "governance": {
            "status":
                "DEMONSTRATION",

            "note":
                "Integrated risk score is an "
                "aggregation layer for portfolio "
                "demonstration. Production deployment "
                "requires model calibration, "
                "independent validation, monitoring, "
                "and approved business policies.",
        },
    }

    # --------------------------------------------------------
    # Print dashboard summary
    # --------------------------------------------------------

    print("## INTEGRATED EXECUTIVE DASHBOARD SUMMARY")
    print()

    print(
        f"Overall Risk Score : "
        f"{overall_score:.2f}"
    )

    print(
        f"Risk Tier          : "
        f"{risk_tier}"
    )

    print(
        f"Alert Status       : "
        f"{alert_status}"
    )

    print(
        f"Top Risk Driver    : "
        f"{top_risk_driver}"
    )

    print()

    print("Risk Domain Scores")
    print(
        f"Credit Risk : {credit_score:.2f}"
    )

    print(
        f"Fraud Risk  : {fraud_score:.2f}"
    )

    print(
        f"AML Risk    : {aml_score:.2f}"
    )

    print()

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    print("Saving integrated dashboard...")

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
    # Save CSVs
    # --------------------------------------------------------

    executive_summary.to_csv(
        DASHBOARD_SUMMARY,
        index=False
    )

    components.to_csv(
        RISK_COMPONENTS,
        index=False
    )

    alert_df.to_csv(
        ALERT_SUMMARY,
        index=False
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print()
    print("Verifying outputs...")

    output_files = [
        DASHBOARD_JSON,
        DASHBOARD_SUMMARY,
        RISK_COMPONENTS,
        ALERT_SUMMARY,
    ]

    for path in output_files:

        if not path.exists():
            raise RuntimeError(
                f"Expected output not created:\n{path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Output file is empty:\n{path}"
            )

    print("All output artifacts verified.")
    print()

    print("## Dashboard output artifacts")
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
        f"Risk components: "
        f"{RISK_COMPONENTS}"
    )

    print(
        f"Alert summary  : "
        f"{ALERT_SUMMARY}"
    )

    print()
    print(
        "Integrated executive dashboard "
        "export completed successfully."
    )


if __name__ == "__main__":
    main()