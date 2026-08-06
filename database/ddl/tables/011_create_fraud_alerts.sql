/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 011_create_fraud_alerts.sql
Author      : Vivek Pandey
Description : Fraud alerts generated from ML model and business rules.
Depends On  : 009_create_transactions.sql
===============================================================================
*/

CREATE TABLE IF NOT EXISTS risk.fraud_alerts
(
    fraud_alert_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    transaction_id            BIGINT NOT NULL,

    alert_timestamp           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    detection_source          VARCHAR(20)
        CHECK (detection_source IN
        ('ML','Rule Engine','Hybrid')),

    fraud_score               NUMERIC(5,4)
        CHECK (fraud_score BETWEEN 0 AND 1),

    risk_level                VARCHAR(20)
        CHECK (risk_level IN
        ('Low','Medium','High','Critical')),

    alert_status              VARCHAR(20)
        DEFAULT 'Open'
        CHECK (alert_status IN
        ('Open','Investigating','Confirmed Fraud',
         'False Positive','Closed')),

    assigned_to               VARCHAR(100),

    investigation_notes       TEXT,

    resolved_at               TIMESTAMP,

    created_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_fraud_transaction
        FOREIGN KEY(transaction_id)
        REFERENCES banking.transactions(transaction_id)
);