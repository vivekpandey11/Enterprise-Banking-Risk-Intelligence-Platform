/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 008_create_devices.sql
Author      : Vivek Pandey
Description : Stores customer device information used for fraud detection.
Depends On  : None
===============================================================================
*/

CREATE TABLE IF NOT EXISTS banking.devices
(
    device_id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    device_fingerprint          VARCHAR(255) NOT NULL UNIQUE,

    device_type                 VARCHAR(30)
        CHECK (device_type IN ('Mobile','Laptop','Desktop','ATM','POS')),

    operating_system            VARCHAR(100),

    browser_name                VARCHAR(100),

    browser_version             VARCHAR(50),

    ip_address                  INET,

    trusted_device              BOOLEAN DEFAULT FALSE,

    first_seen_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    last_seen_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE banking.devices IS
'Stores customer device fingerprints for fraud detection and risk analysis.';