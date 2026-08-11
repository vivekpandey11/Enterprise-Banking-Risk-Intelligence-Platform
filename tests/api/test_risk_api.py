from fastapi.testclient import TestClient

from src.api.risk_api import app


client = TestClient(app)


# ============================================================
# HEALTH CHECK
# ============================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "enterprise-risk-intelligence-api"
    assert data["version"] == "1.0.0"
    assert "artifacts_available" in data


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["version"] == "1.0.0"

    assert "credit-risk" in data["modules"]
    assert "fraud-detection" in data["modules"]
    assert "aml-monitoring" in data["modules"]
    assert "integrated-risk" in data["modules"]

    assert "/health" in data["endpoints"]
    assert "/model-status" in data["endpoints"]
    assert "/risk-assessment" in data["endpoints"]


# ============================================================
# MODEL STATUS
# ============================================================

def test_model_status():

    response = client.get("/model-status")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["credit_risk"] == "available"
    assert data["fraud_detection"] == "available"
    assert data["aml_monitoring"] == "available"
    assert data["integrated_risk_engine"] == "available"

    assert "assessment_artifact" in data
    assert "summary_artifact" in data
    assert "current_risk_tier" in data


# ============================================================
# RISK ASSESSMENT - LOW RISK
# ============================================================

def test_risk_assessment_low_risk():

    payload = {
        "customer_id": "TEST-LOW-001",
        "transaction_amount": 1000,
        "country_from": "US",
        "country_to": "US",
        "currency": "USD",
    }

    response = client.post(
        "/risk-assessment",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "request" in data
    assert "risk_assessment" in data
    assert "business_decision" in data
    assert "metadata" in data

    assert data["request"]["customer_id"] == "TEST-LOW-001"
    assert data["request"]["transaction_amount"] == 1000
    assert data["request"]["cross_border"] is False
    assert data["request"]["high_value_transaction"] is False

    assert "credit_risk" in data["risk_assessment"]
    assert "fraud_risk" in data["risk_assessment"]
    assert "aml_risk" in data["risk_assessment"]
    assert "integrated_risk" in data["risk_assessment"]


# ============================================================
# RISK ASSESSMENT - HIGH VALUE + CROSS BORDER
# ============================================================

def test_risk_assessment_high_value_cross_border():

    payload = {
        "customer_id": "TEST-HIGH-001",
        "transaction_amount": 15000,
        "country_from": "US",
        "country_to": "HK",
        "currency": "USD",
    }

    response = client.post(
        "/risk-assessment",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    request_data = data["request"]

    assert request_data["transaction_amount"] == 15000
    assert request_data["cross_border"] is True
    assert request_data["high_value_transaction"] is True

    risk = data["risk_assessment"]

    assert "aml_risk" in risk

    aml = risk["aml_risk"]

    assert "score" in aml
    assert "risk_segment" in aml
    assert "decision" in aml


# ============================================================
# VALIDATION - NEGATIVE TRANSACTION
# ============================================================

def test_negative_transaction_amount_rejected():

    payload = {
        "customer_id": "TEST-INVALID-001",
        "transaction_amount": -100,
        "country_from": "US",
        "country_to": "US",
        "currency": "USD",
    }

    response = client.post(
        "/risk-assessment",
        json=payload,
    )

    assert response.status_code == 422


# ============================================================
# VALIDATION - INVALID DATA TYPE
# ============================================================

def test_invalid_transaction_amount_type():

    payload = {
        "customer_id": "TEST-INVALID-002",
        "transaction_amount": "invalid",
        "country_from": "US",
        "country_to": "US",
        "currency": "USD",
    }

    response = client.post(
        "/risk-assessment",
        json=payload,
    )

    assert response.status_code == 422


# ============================================================
# RESPONSE STRUCTURE
# ============================================================

def test_risk_assessment_response_structure():

    payload = {
        "customer_id": "TEST-STRUCTURE-001",
        "transaction_amount": 5000,
        "country_from": "US",
        "country_to": "GB",
        "currency": "USD",
    }

    response = client.post(
        "/risk-assessment",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    risk = data["risk_assessment"]

    credit = risk["credit_risk"]
    fraud = risk["fraud_risk"]
    aml = risk["aml_risk"]
    integrated = risk["integrated_risk"]

    assert "score" in credit
    assert "risk_segment" in credit
    assert "decision" in credit

    assert "score" in fraud
    assert "risk_segment" in fraud
    assert "decision" in fraud

    assert "score" in aml
    assert "risk_segment" in aml
    assert "decision" in aml

    assert "overall_score" in integrated
    assert "risk_tier" in integrated
    assert "alert_status" in integrated
    assert "top_risk_driver" in integrated

    business = data["business_decision"]

    assert "approved_for_demo" in business
    assert "requires_review" in business


# ============================================================
# TEST COMPLETE
# ============================================================

if __name__ == "__main__":
    print("Run tests using:")
    print("pytest -v tests/api/test_risk_api.py")