/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 001_create_customers.sql
Author      : Vivek Pandey
Description : Creates the master customer table.
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.customers
(
    customer_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    customer_code      VARCHAR(20) NOT NULL UNIQUE,

    first_name         VARCHAR(100) NOT NULL,

    last_name          VARCHAR(100) NOT NULL,

    date_of_birth      DATE NOT NULL,

    gender             VARCHAR(10)
        CHECK (gender IN ('Male','Female','Other')),

    email              VARCHAR(255) UNIQUE,

    mobile_number      VARCHAR(20) UNIQUE,

    pan_number         VARCHAR(10) UNIQUE,

    customer_type      VARCHAR(20)
        DEFAULT 'Individual'
        CHECK (customer_type IN ('Individual','Corporate')),

    risk_category      VARCHAR(20)
        DEFAULT 'Low'
        CHECK (risk_category IN ('Low','Medium','High')),

    kyc_status         VARCHAR(20)
        DEFAULT 'Pending'
        CHECK (kyc_status IN ('Pending','Verified','Rejected')),

    record_status      VARCHAR(20)
        DEFAULT 'Active'
        CHECK (record_status IN ('Active','Inactive')),

    created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);