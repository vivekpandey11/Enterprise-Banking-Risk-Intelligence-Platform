/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 003_create_accounts.sql
Author      : Vivek Pandey
Description : Creates customer bank accounts table.
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.accounts
(
    account_id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    account_number          VARCHAR(20) NOT NULL UNIQUE,

    customer_id             BIGINT NOT NULL,

    branch_id               BIGINT NOT NULL,

    account_type            VARCHAR(20)
        DEFAULT 'Savings'
        CHECK (account_type IN ('Savings','Current','Salary','Fixed Deposit')),

    account_status          VARCHAR(20)
        DEFAULT 'Active'
        CHECK (account_status IN ('Active','Inactive','Blocked','Closed')),

    opening_balance         NUMERIC(18,2)
        DEFAULT 0
        CHECK (opening_balance >= 0),

    current_balance         NUMERIC(18,2)
        DEFAULT 0
        CHECK (current_balance >= 0),

    currency_code           CHAR(3)
        DEFAULT 'INR',

    opened_date             DATE NOT NULL,

    created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_accounts_customer
        FOREIGN KEY (customer_id)
        REFERENCES banking.customers(customer_id),

    CONSTRAINT fk_accounts_branch
        FOREIGN KEY (branch_id)
        REFERENCES banking.branches(branch_id)
);

COMMENT ON TABLE banking.accounts IS
'Customer bank accounts.';