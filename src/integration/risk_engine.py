from pathlib import Path
import json

# ============================================================
# ENTERPRISE INTEGRATED RISK ENGINE
# Credit Risk + Fraud Risk + AML Risk
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGING_DIR = PROJECT_ROOT / "data" / "staging"

CREDIT_DIR = STAGING_DIR / "credit_risk"
FRAUD_DIR = STAGING_DIR / "fraud"
AML_DIR = STAGING_DIR / "aml"

OUTPUT_DIR = STAGING_DIR / "integrated_risk"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "integrated_risk_assessment.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "integrated_risk_summary.json"


# ------------------------------------------------------------
# Candidate artifact paths
# ------------------------------------------------------------

CREDIT_CANDIDATES = [
    CREDIT_DIR / "credit_risk_demo" / "credit_risk_demo_prediction.json",
    CREDIT_DIR / "credit_risk_demo_prediction.json",
    CREDIT_DIR / "credit_risk_final" / "credit_risk_final_evaluation.json",
]

FRAUD_CANDIDATES = [
    FRAUD_DIR / "fraud_demo_prediction.json",
    FRAUD_DIR / "fraud_demo" / "fraud_demo_prediction.json",
]

AML_CANDIDATES = [
    AML_DIR / "aml_demo" / "aml_demo_prediction.json",
    AML_DIR / "aml_demo_prediction.json",
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_first_existing(candidates, artifact_name):
    for path in candidates:
        if path.exists():
            print(f"[FOUND] {artifact_name}: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Could not find {artifact_name}.\n"
        f"Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return float(value)

        return float(value)

    except (TypeError, ValueError):
        return default


def first_value(data, keys, default=None):
    """
    Return the first available value from a dictionary.
    """

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def normalize_probability(value):
    """
    Converts probability into percentage-free 0-1 form.

    Examples:
        0.25       -> 0.25
        25         -> 0.25
        0.25%      -> 0.0025
    """

    if isinstance(value, str):
        value = value.replace("%", "").strip()

    value = safe_float(value)

    if value > 1:
        value = value / 100

    return max(0.0, min(1.0, value))


def normalize_risk_score(value):
    """
    Converts arbitrary risk score into 0-100.
    """

    value = safe_float(value)

    if value <= 1:
        value *= 100

    return max(0.0, min(100.0, value))


# ------------------------------------------------------------
# Extract Credit Risk
# ------------------------------------------------------------

def extract_credit_risk(data):

    probability = first_value(
        data,
        [
            "credit_risk_probability",
            "default_probability",
            "probability",
            "risk_probability",
            "pd",
        ],
        0,
    )

    probability = normalize_probability(probability)

    score = normalize_risk_score(
        first_value(
            data,
            [
                "credit_risk_score",
                "risk_score",
                "score",
            ],
            probability * 100,
        )
    )

    risk_segment = first_value(
        data,
        [
            "risk_segment",
            "credit_risk_segment",
            "risk_level",
        ],
        "UNKNOWN",
    )

    decision = first_value(
        data,
        [
            "credit_risk_decision",
            "decision",
            "prediction",
        ],
        "UNKNOWN",
    )

    return {
        "probability": probability,
        "risk_score": score,
        "risk_segment": str(risk_segment),
        "decision": str(decision),
    }


# ------------------------------------------------------------
# Extract Fraud Risk
# ------------------------------------------------------------

def extract_fraud_risk(data):

    probability = first_value(
        data,
        [
            "fraud_probability",
            "fraud_score",
            "probability",
            "risk_probability",
        ],
        0,
    )

    probability = normalize_probability(probability)

    score = normalize_risk_score(
        first_value(
            data,
            [
                "fraud_risk_score",
                "risk_score",
                "score",
            ],
            probability * 100,
        )
    )

    risk_segment = first_value(
        data,
        [
            "risk_segment",
            "fraud_risk_segment",
            "risk_level",
        ],
        "UNKNOWN",
    )

    decision = first_value(
        data,
        [
            "fraud_decision",
            "decision",
            "prediction",
        ],
        "UNKNOWN",
    )

    return {
        "probability": probability,
        "risk_score": score,
        "risk_segment": str(risk_segment),
        "decision": str(decision),
    }


# ------------------------------------------------------------
# Extract AML Risk
# ------------------------------------------------------------

def extract_aml_risk(data):

    probability = first_value(
        data,
        [
            "aml_probability",
            "probability",
            "risk_probability",
        ],
        0,
    )

    probability = normalize_probability(probability)

    score = normalize_risk_score(
        first_value(
            data,
            [
                "aml_risk_score",
                "risk_score",
                "score",
            ],
            probability * 100,
        )
    )

    risk_segment = first_value(
        data,
        [
            "risk_segment",
            "aml_risk_segment",
            "risk_level",
        ],
        "UNKNOWN",
    )

    decision = first_value(
        data,
        [
            "aml_decision",
            "decision",
            "prediction",
        ],
        "UNKNOWN",
    )

    alert_status = first_value(
        data,
        [
            "alert_status",
            "alert",
        ],
        "NO_ALERT",
    )

    return {
        "probability": probability,
        "risk_score": score,
        "risk_segment": str(risk_segment),
        "decision": str(decision),
        "alert_status": str(alert_status),
    }


# ------------------------------------------------------------
# Overall Risk Calculation
# ------------------------------------------------------------

def calculate_integrated_risk(
    credit_score,
    fraud_score,
    aml_score,
):
    """
    Enterprise risk weighting:

        Credit Risk = 35%
        Fraud Risk  = 35%
        AML Risk    = 30%
    """

    weighted_score = (
        credit_score * 0.35
        + fraud_score * 0.35
        + aml_score * 0.30
    )

    return round(weighted_score, 4)


def classify_risk(score):

    if score >= 75:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 25:
        return "MEDIUM"

    return "LOW"


def determine_alert(
    risk_score,
    credit,
    fraud,
    aml,
):

    if risk_score >= 75:
        return "CRITICAL_ALERT"

    if risk_score >= 50:
        return "HIGH_RISK_ALERT"

    if fraud["risk_score"] >= 50:
        return "FRAUD_ALERT"

    if aml["risk_score"] >= 50:
        return "AML_ALERT"

    if credit["risk_score"] >= 50:
        return "CREDIT_RISK_ALERT"

    return "NO_ALERT"


# ============================================================
# MAIN
# ============================================================

def main():

    print("# ENTERPRISE INTEGRATED RISK ENGINE")
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print()

    # --------------------------------------------------------
    # Load module outputs
    # --------------------------------------------------------

    print("Loading Credit Risk assessment...")

    credit_data = load_first_existing(
        CREDIT_CANDIDATES,
        "Credit Risk prediction",
    )

    print()

    print("Loading Fraud assessment...")

    fraud_data = load_first_existing(
        FRAUD_CANDIDATES,
        "Fraud prediction",
    )

    print()

    print("Loading AML assessment...")

    aml_data = load_first_existing(
        AML_CANDIDATES,
        "AML prediction",
    )

    print()

    # --------------------------------------------------------
    # Extract normalized risk information
    # --------------------------------------------------------

    credit = extract_credit_risk(
        credit_data
    )

    fraud = extract_fraud_risk(
        fraud_data
    )

    aml = extract_aml_risk(
        aml_data
    )

    # --------------------------------------------------------
    # Integrated score
    # --------------------------------------------------------

    overall_score = calculate_integrated_risk(
        credit["risk_score"],
        fraud["risk_score"],
        aml["risk_score"],
    )

    risk_tier = classify_risk(
        overall_score
    )

    alert_status = determine_alert(
        overall_score,
        credit,
        fraud,
        aml,
    )

    # --------------------------------------------------------
    # Risk drivers
    # --------------------------------------------------------

    risk_components = {
        "credit_risk": credit["risk_score"],
        "fraud_risk": fraud["risk_score"],
        "aml_risk": aml["risk_score"],
    }

    sorted_components = sorted(
        risk_components.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top_risk_driver = (
        sorted_components[0][0]
        if sorted_components
        else "UNKNOWN"
    )

    # --------------------------------------------------------
    # Final assessment
    # --------------------------------------------------------

    assessment = {

        "platform": {
            "name": (
                "Enterprise Banking Risk "
                "Intelligence Platform"
            ),
            "engine": "Integrated Risk Engine",
            "version": "1.0",
        },

        "risk_weights": {
            "credit_risk": 0.35,
            "fraud_risk": 0.35,
            "aml_risk": 0.30,
        },

        "credit_risk": credit,

        "fraud_risk": fraud,

        "aml_risk": aml,

        "integrated_risk": {
            "overall_risk_score": overall_score,
            "risk_tier": risk_tier,
            "alert_status": alert_status,
            "top_risk_driver": top_risk_driver,
        },

        "governance": {
            "note": (
                "Integrated score is a demonstration "
                "risk aggregation layer. Individual "
                "model outputs should be validated "
                "against production calibration and "
                "business policy before deployment."
            )
        },
    }

    # --------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------

    summary = {
        "overall_risk_score": overall_score,
        "risk_tier": risk_tier,
        "alert_status": alert_status,
        "top_risk_driver": top_risk_driver,
        "credit_risk_score": credit["risk_score"],
        "fraud_risk_score": fraud["risk_score"],
        "aml_risk_score": aml["risk_score"],
    }

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("## INTEGRATED RISK ASSESSMENT")
    print()

    print(
        f"Credit Risk Score : "
        f"{credit['risk_score']:.2f}"
    )

    print(
        f"Fraud Risk Score  : "
        f"{fraud['risk_score']:.2f}"
    )

    print(
        f"AML Risk Score    : "
        f"{aml['risk_score']:.2f}"
    )

    print()

    print(
        f"Overall Risk Score: "
        f"{overall_score:.2f}"
    )

    print(
        f"Risk Tier         : "
        f"{risk_tier}"
    )

    print(
        f"Alert Status      : "
        f"{alert_status}"
    )

    print(
        f"Top Risk Driver   : "
        f"{top_risk_driver}"
    )

    print()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("Saving integrated risk assessment...")

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            assessment,
            f,
            indent=2,
            default=str,
        )

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print()
    print("Verifying outputs...")

    for path in [
        OUTPUT_JSON,
        OUTPUT_SUMMARY,
    ]:

        if not path.exists():
            raise RuntimeError(
                f"Expected output not created: {path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Output file is empty: {path}"
            )

    print("All output artifacts verified.")
    print()

    print("## Output artifacts")
    print()

    print(
        f"Integrated assessment : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Risk summary          : "
        f"{OUTPUT_SUMMARY}"
    )

    print()
    print(
        "Enterprise integrated risk "
        "engine completed successfully."
    )


if __name__ == "__main__":
    main()