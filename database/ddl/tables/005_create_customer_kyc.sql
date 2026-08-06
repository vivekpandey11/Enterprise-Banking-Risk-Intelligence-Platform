/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 005_create_customer_kyc.sql
Author      : Vivek Pandey
Description : Stores KYC details for customers.
Depends On  : 001_create_customers.sql
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.customer_kyc
(
    kyc_id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    customer_id             BIGINT NOT NULL UNIQUE,

    pan_number              VARCHAR(10) NOT NULL UNIQUE,

    aadhaar_number          VARCHAR(12) UNIQUE,

    pan_verified            BOOLEAN DEFAULT FALSE,

    aadhaar_verified        BOOLEAN DEFAULT FALSE,

    kyc_completion_date     DATE,

    kyc_expiry_date         DATE,

    risk_rating             VARCHAR(20)
        DEFAULT 'Low'
        CHECK (risk_rating IN ('Low','Medium','High')),

    pep_flag                BOOLEAN DEFAULT FALSE,

    sanctions_flag          BOOLEAN DEFAULT FALSE,

    created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_customer_kyc
        FOREIGN KEY (customer_id)
        REFERENCES banking.customers(customer_id)
        ON DELETE CASCADE
);

COMMENT ON TABLE banking.customer_kyc IS
'Stores KYC verification and regulatory compliance details for customers.';