from pathlib import Path
import json
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ============================================================
# ENTERPRISE RISK INTELLIGENCE API
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTEGRATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "integrated_risk"
)

ASSESSMENT_FILE = (
    INTEGRATED_DIR
    / "integrated_risk_assessment.json"
)

SUMMARY_FILE = (
    INTEGRATED_DIR
    / "integrated_risk_summary.json"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Enterprise Banking Risk Intelligence API",
    description=(
        "Unified Credit Risk, Fraud Detection and "
        "AML Risk Assessment API."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class RiskRequest(BaseModel):

    customer_id: str = Field(
        default="DEMO-CUSTOMER",
        description="Customer identifier"
    )

    transaction_amount: float = Field(
        default=1000.0,
        ge=0,
        description="Transaction amount"
    )

    country_from: str = Field(
        default="US",
        description="Origin country"
    )

    country_to: str = Field(
        default="US",
        description="Destination country"
    )

    currency: str = Field(
        default="USD",
        description="Transaction currency"
    )


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return default


def clean_decision(value):

    """
    Converts old dictionary-string decisions into
    clean structured API decisions.
    """

    if isinstance(value, dict):
        return value

    if value is None:
        return "UNKNOWN"

    value = str(value)

    # Credit model dictionary string
    if "credit_decision" in value:

        if "'credit_decision': 'LOW_RISK'" in value:
            return "LOW_RISK"

        if "'credit_decision': 'REVIEW'" in value:
            return "REVIEW"

        if "'credit_decision': 'HIGH_RISK'" in value:
            return "HIGH_RISK"

    # Fraud model dictionary string
    if "fraud_decision" in value:

        if "'fraud_decision': 'LOW_RISK'" in value:
            return "LOW_RISK"

        if "'fraud_decision': 'REVIEW'" in value:
            return "REVIEW"

        if "'fraud_decision': 'HIGH_RISK'" in value:
            return "HIGH_RISK"

    return value


def extract_probability(
    data,
    probability_keys
):

    for key in probability_keys:

        if key in data:

            return safe_float(
                data.get(key)
            )

    return 0.0


def normalize_risk_segment(
    value,
    default="UNKNOWN"
):

    if value is None:
        return default

    value = str(value).strip()

    mapping = {
        "low": "LOW RISK",
        "low risk": "LOW RISK",
        "medium": "MEDIUM RISK",
        "medium risk": "MEDIUM RISK",
        "high": "HIGH RISK",
        "high risk": "HIGH RISK",
        "critical": "CRITICAL RISK",
        "critical risk": "CRITICAL RISK",
    }

    return mapping.get(
        value.lower(),
        value
    )


def get_risk_data():

    assessment = load_json(
        ASSESSMENT_FILE
    )

    summary = load_json(
        SUMMARY_FILE
    )

    return assessment, summary


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "service":
            "enterprise-risk-intelligence-api",

        "version":
            "1.0.0",

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "artifacts_available":
            ASSESSMENT_FILE.exists()
            and SUMMARY_FILE.exists(),
    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/model-status")
def model_status():

    assessment, summary = (
        get_risk_data()
    )

    return {

        "status": "ready",

        "credit_risk":
            "available",

        "fraud_detection":
            "available",

        "aml_monitoring":
            "available",

        "integrated_risk_engine":
            "available",

        "assessment_artifact":
            str(ASSESSMENT_FILE),

        "summary_artifact":
            str(SUMMARY_FILE),

        "current_risk_tier":
            summary.get(
                "risk_tier",
                assessment.get(
                    "risk_tier",
                    "UNKNOWN"
                )
            ),
    }


# ============================================================
# RISK ASSESSMENT
# ============================================================

@app.post("/risk-assessment")
def risk_assessment(
    request: RiskRequest
):

    assessment, summary = (
        get_risk_data()
    )

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

    # ========================================================
    # REQUEST BUSINESS FEATURES
    # ========================================================

    country_from = (
        request.country_from
        .strip()
        .upper()
    )

    country_to = (
        request.country_to
        .strip()
        .upper()
    )

    currency = (
        request.currency
        .strip()
        .upper()
    )

    cross_border = (
        country_from != country_to
    )

    high_value = (
        request.transaction_amount >= 10000
    )

    # ========================================================
    # CREDIT RISK
    # ========================================================

    credit_probability = extract_probability(
        credit,
        [
            "default_probability",
            "credit_probability",
            "probability",
            "risk_probability",
        ]
    )

    credit_prediction = credit.get(
        "default_prediction",
        credit.get(
            "prediction",
            0
        )
    )

    credit_segment = normalize_risk_segment(
        credit.get(
            "risk_segment",
            "UNKNOWN"
        )
    )

    credit_decision = clean_decision(
        credit.get(
            "decision",
            credit.get(
                "credit_decision",
                "UNKNOWN"
            )
        )
    )

    # ========================================================
    # FRAUD RISK
    # ========================================================

    fraud_probability = extract_probability(
        fraud,
        [
            "fraud_probability",
            "probability",
            "risk_probability",
        ]
    )

    fraud_prediction = fraud.get(
        "fraud_prediction",
        fraud.get(
            "prediction",
            0
        )
    )

    fraud_segment = normalize_risk_segment(
        fraud.get(
            "risk_segment",
            "UNKNOWN"
        )
    )

    fraud_decision = clean_decision(
        fraud.get(
            "decision",
            fraud.get(
                "fraud_decision",
                "UNKNOWN"
            )
        )
    )

    # ========================================================
    # AML RISK
    # ========================================================

    aml_probability = extract_probability(
        aml,
        [
            "aml_probability",
            "probability",
            "risk_probability",
        ]
    )

    aml_prediction = aml.get(
        "aml_prediction",
        aml.get(
            "prediction",
            0
        )
    )

    aml_segment = normalize_risk_segment(
        aml.get(
            "risk_segment",
            "UNKNOWN"
        )
    )

    aml_decision = clean_decision(
        aml.get(
            "decision",
            aml.get(
                "aml_decision",
                "UNKNOWN"
            )
        )
    )

    # ========================================================
    # RISK SCORES
    # ========================================================

    credit_score = (
        credit_probability * 100
    )

    fraud_score = (
        fraud_probability * 100
    )

    aml_score = (
        aml_probability * 100
    )

    # ========================================================
    # INTEGRATED RISK
    # ========================================================

    saved_overall_score = safe_float(
        integrated.get(
            "overall_risk_score",
            summary.get(
                "overall_risk_score",
                0
            )
        )
    )

    saved_risk_tier = integrated.get(
        "risk_tier",
        summary.get(
            "risk_tier",
            "UNKNOWN"
        )
    )

    saved_alert_status = integrated.get(
        "alert_status",
        summary.get(
            "alert_status",
            "NO_ALERT"
        )
    )

    saved_top_driver = integrated.get(
        "top_risk_driver",
        summary.get(
            "top_risk_driver",
            "UNKNOWN"
        )
    )

    # --------------------------------------------------------
    # Calculate a request-aware score.
    #
    # This keeps the API useful for demonstrations while
    # retaining the saved model outputs.
    # --------------------------------------------------------

    transaction_factor = 0.0

    if high_value:
        transaction_factor += 10.0

    if cross_border:
        transaction_factor += 5.0

    request_aml_score = min(
        100.0,
        aml_score + transaction_factor
    )

    overall_score = max(
        credit_score,
        fraud_score,
        request_aml_score
    )

    # --------------------------------------------------------
    # Risk tier
    # --------------------------------------------------------

    if overall_score >= 75:

        risk_tier = "CRITICAL"

    elif overall_score >= 50:

        risk_tier = "HIGH"

    elif overall_score >= 25:

        risk_tier = "MEDIUM"

    else:

        risk_tier = "LOW"

    # --------------------------------------------------------
    # Alert status
    # --------------------------------------------------------

    if (
        risk_tier in (
            "HIGH",
            "CRITICAL"
        )
        or credit_decision in (
            "HIGH_RISK",
            "REVIEW"
        )
        or fraud_decision in (
            "HIGH_RISK",
            "REVIEW"
        )
        or aml_decision in (
            "HIGH_RISK",
            "REVIEW"
        )
    ):

        alert_status = "REVIEW_REQUIRED"

    else:

        alert_status = "NO_ALERT"

    # --------------------------------------------------------
    # Top risk driver
    # --------------------------------------------------------

    risk_components = {

        "credit_risk":
            credit_score,

        "fraud_risk":
            fraud_score,

        "aml_risk":
            request_aml_score,
    }

    top_risk_driver = max(
        risk_components,
        key=risk_components.get
    )

    # ========================================================
    # BUSINESS DECISION
    # ========================================================

    approved_for_demo = (
        risk_tier == "LOW"
        and alert_status == "NO_ALERT"
    )

    requires_review = not approved_for_demo

    # ========================================================
    # RESPONSE
    # ========================================================

    response = {

        "request": {

            "customer_id":
                request.customer_id,

            "transaction_amount":
                request.transaction_amount,

            "country_from":
                country_from,

            "country_to":
                country_to,

            "currency":
                currency,

            "cross_border":
                cross_border,

            "high_value_transaction":
                high_value,
        },

        "risk_assessment": {

            "credit_risk": {

                "score":
                    round(
                        credit_score,
                        6
                    ),

                "probability":
                    round(
                        credit_probability,
                        6
                    ),

                "risk_segment":
                    credit_segment,

                "decision":
                    credit_decision,

                "prediction":
                    int(
                        safe_float(
                            credit_prediction
                        )
                    ),
            },

            "fraud_risk": {

                "score":
                    round(
                        fraud_score,
                        6
                    ),

                "probability":
                    round(
                        fraud_probability,
                        6
                    ),

                "risk_segment":
                    fraud_segment,

                "decision":
                    fraud_decision,

                "prediction":
                    int(
                        safe_float(
                            fraud_prediction
                        )
                    ),
            },

            "aml_risk": {

                "score":
                    round(
                        request_aml_score,
                        6
                    ),

                "model_probability":
                    round(
                        aml_probability,
                        10
                    ),

                "risk_segment":
                    aml_segment,

                "decision":
                    aml_decision,

                "prediction":
                    int(
                        safe_float(
                            aml_prediction
                        )
                    ),

                "transaction_factors": {

                    "cross_border":
                        cross_border,

                    "high_value_transaction":
                        high_value,

                    "transaction_factor":
                        transaction_factor,
                },
            },

            "integrated_risk": {

                "overall_score":
                    round(
                        overall_score,
                        6
                    ),

                "risk_tier":
                    risk_tier,

                "alert_status":
                    alert_status,

                "top_risk_driver":
                    top_risk_driver,

                "saved_engine_score":
                    round(
                        saved_overall_score,
                        6
                    ),

                "saved_engine_risk_tier":
                    saved_risk_tier,

                "saved_engine_alert_status":
                    saved_alert_status,

                "saved_engine_top_risk_driver":
                    saved_top_driver,
            },
        },

        "business_decision": {

            "approved_for_demo":
                approved_for_demo,

            "requires_review":
                requires_review,

            "reason":
                (
                    "Transaction is LOW risk "
                    "and requires no review."
                    if approved_for_demo
                    else
                    "Transaction requires "
                    "risk review."
                ),
        },

        "metadata": {

            "generated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "api_version":
                "1.0.0",

            "environment":
                "demonstration",
        },
    }

    return response


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            "Enterprise Banking Risk Intelligence API",

        "version":
            "1.0.0",

        "status":
            "running",

        "modules": [

            "credit-risk",

            "fraud-detection",

            "aml-monitoring",

            "integrated-risk",
        ],

        "endpoints": [

            "/",

            "/health",

            "/model-status",

            "/risk-assessment",

            "/docs",
        ],
    }