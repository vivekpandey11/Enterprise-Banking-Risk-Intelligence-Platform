/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 010_create_credit_scores.sql
Author      : Vivek Pandey
Description : Customer credit bureau and internal credit score data.
Depends On  : 001_create_customers.sql
===============================================================================
*/

CREATE TABLE IF NOT EXISTS risk.credit_scores
(
    credit_score_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    customer_id                BIGINT NOT NULL,

    bureau_name                VARCHAR(50)
        CHECK (bureau_name IN
        ('CIBIL','Experian','Equifax','CRIF')),

    bureau_score               INTEGER
        CHECK (bureau_score BETWEEN 300 AND 900),

    internal_score             NUMERIC(5,2),

    default_probability        NUMERIC(5,4),

    score_date                 DATE NOT NULL,

    score_version              VARCHAR(20),

    created_at                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_credit_customer
        FOREIGN KEY(customer_id)
        REFERENCES banking.customers(customer_id)
);