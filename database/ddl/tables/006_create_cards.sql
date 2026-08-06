/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 006_create_cards.sql
Author      : Vivek Pandey
Description : Stores customer debit and credit card details.
Depends On  : 003_create_accounts.sql
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.cards
(
    card_id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    account_id               BIGINT NOT NULL,

    card_number              VARCHAR(19) NOT NULL UNIQUE,

    card_type                VARCHAR(20)
        CHECK (card_type IN ('Debit','Credit')),

    card_network             VARCHAR(20)
        CHECK (card_network IN ('Visa','Mastercard','RuPay','Amex')),

    issue_date               DATE NOT NULL,

    expiry_date              DATE NOT NULL,

    card_status              VARCHAR(20)
        DEFAULT 'Active'
        CHECK (card_status IN ('Active','Blocked','Expired','Closed')),

    daily_transaction_limit  NUMERIC(18,2)
        DEFAULT 50000
        CHECK (daily_transaction_limit >= 0),

    international_enabled    BOOLEAN DEFAULT FALSE,

    contactless_enabled      BOOLEAN DEFAULT TRUE,

    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_cards_account
        FOREIGN KEY (account_id)
        REFERENCES banking.accounts(account_id)
        ON DELETE CASCADE
);

COMMENT ON TABLE banking.cards IS
'Stores debit and credit card information linked to customer accounts.';