/*
===============================================================================
Enterprise Banking Risk Intelligence Platform (EBRIP)
File        : 012_create_aml_rules.sql
Author      : Vivek Pandey
Description : AML rules configuration master table.
===============================================================================
*/

CREATE TABLE IF NOT EXISTS risk.aml_rules
(
    rule_id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    rule_code            VARCHAR(30) UNIQUE NOT NULL,

    rule_name            VARCHAR(200) NOT NULL,

    description          TEXT,

    threshold_amount     NUMERIC(18,2),

    threshold_count      INTEGER,

    time_window_minutes  INTEGER,

    severity             VARCHAR(20)
        CHECK (severity IN ('Low','Medium','High','Critical')),

    is_active            BOOLEAN DEFAULT TRUE,

    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE risk.aml_rules
IS 'AML monitoring rules used by the compliance engine.';