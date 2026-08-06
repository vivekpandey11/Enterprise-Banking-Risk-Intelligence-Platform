/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 007_create_merchants.sql
Author      : Vivek Pandey
Description : Stores merchant master information.
Depends On  : None
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.merchants
(
    merchant_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    merchant_code          VARCHAR(20) NOT NULL UNIQUE,

    merchant_name          VARCHAR(200) NOT NULL,

    merchant_category      VARCHAR(100) NOT NULL,

    city                   VARCHAR(100),

    state                  VARCHAR(100),

    country                VARCHAR(100) DEFAULT 'India',

    risk_level             VARCHAR(20)
        DEFAULT 'Low'
        CHECK (risk_level IN ('Low','Medium','High')),

    merchant_status        VARCHAR(20)
        DEFAULT 'Active'
        CHECK (merchant_status IN ('Active','Inactive')),

    created_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE banking.merchants IS
'Master table containing merchant information for transaction analysis.';