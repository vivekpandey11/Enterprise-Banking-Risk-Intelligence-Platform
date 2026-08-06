/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 009_create_transactions.sql
Author      : Vivek Pandey
Description : Stores all banking transactions.
Depends On  :
  - 003_create_accounts.sql
  - 006_create_cards.sql
  - 007_create_merchants.sql
  - 008_create_devices.sql
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.transactions
(
    transaction_id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    transaction_reference       VARCHAR(40) NOT NULL UNIQUE,

    account_id                  BIGINT NOT NULL,

    card_id                     BIGINT,

    merchant_id                 BIGINT,

    device_id                   BIGINT,

    transaction_timestamp       TIMESTAMP NOT NULL,

    transaction_type            VARCHAR(30)
        CHECK (transaction_type IN
        ('Purchase','Cash Withdrawal','Cash Deposit','Transfer',
         'Bill Payment','UPI','NEFT','RTGS','IMPS')),

    transaction_channel         VARCHAR(20)
        CHECK (transaction_channel IN
        ('ATM','POS','Mobile','Internet Banking','Branch')),

    debit_credit                CHAR(1)
        CHECK (debit_credit IN ('D','C')),

    transaction_amount          NUMERIC(18,2)
        CHECK (transaction_amount > 0),

    currency_code               CHAR(3) DEFAULT 'INR',

    transaction_status          VARCHAR(20)
        DEFAULT 'Success'
        CHECK (transaction_status IN
        ('Success','Failed','Pending','Reversed')),

    available_balance           NUMERIC(18,2),

    remarks                     VARCHAR(500),

    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_transaction_account
        FOREIGN KEY(account_id)
        REFERENCES banking.accounts(account_id),

    CONSTRAINT fk_transaction_card
        FOREIGN KEY(card_id)
        REFERENCES banking.cards(card_id),

    CONSTRAINT fk_transaction_merchant
        FOREIGN KEY(merchant_id)
        REFERENCES banking.merchants(merchant_id),

    CONSTRAINT fk_transaction_device
        FOREIGN KEY(device_id)
        REFERENCES banking.devices(device_id)
);

COMMENT ON TABLE banking.transactions
IS 'Stores all financial transactions for analytics, fraud detection and AML.';