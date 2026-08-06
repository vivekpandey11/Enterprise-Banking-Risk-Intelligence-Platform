/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 004_create_customer_addresses.sql
Author      : Vivek Pandey
Description : Stores customer addresses.
Depends On  : 001_create_customers.sql
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.customer_addresses
(
    address_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    customer_id         BIGINT NOT NULL,

    address_type        VARCHAR(20)
        CHECK(address_type IN ('Permanent','Current','Office')),

    address_line_1      VARCHAR(255) NOT NULL,

    address_line_2      VARCHAR(255),

    city                VARCHAR(100) NOT NULL,

    state               VARCHAR(100) NOT NULL,

    postal_code         VARCHAR(10) NOT NULL,

    country             VARCHAR(100)
        DEFAULT 'India',

    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_customer_address
        FOREIGN KEY(customer_id)
        REFERENCES banking.customers(customer_id)
        ON DELETE CASCADE
);

COMMENT ON TABLE banking.customer_addresses
IS 'Stores permanent/current/office addresses of customers.';