/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 002_create_branches.sql
Author      : Vivek Pandey
Description : Creates the bank branches master table.
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.branches
(
    branch_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    branch_code         VARCHAR(10) NOT NULL UNIQUE,

    branch_name         VARCHAR(150) NOT NULL,

    city                VARCHAR(100) NOT NULL,

    state               VARCHAR(100) NOT NULL,

    region              VARCHAR(100),

    ifsc_code           VARCHAR(11) NOT NULL UNIQUE,

    branch_status       VARCHAR(20)
        DEFAULT 'Active'
        CHECK (branch_status IN ('Active','Inactive')),

    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE banking.branches IS
'Master table containing bank branch information.';