"""
EBRIP - Transaction Fraud Rule Engine

Deterministic rule-based transaction fraud detection layer.
Evaluates banking transactions using customer, KYC, card,
merchant, device and transaction-risk signals.

Detection source:
    Rule Engine
"""

from pathlib import Path
import os
import sys

import psycopg2
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "ebrip"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}


# ---------------------------------------------------------------------------
# FRAUD SCORING QUERY
# ---------------------------------------------------------------------------

SCORE_SQL = """
SELECT
    t.transaction_id,
    t.transaction_amount,
    t.transaction_type,
    t.transaction_channel,
    t.transaction_status,
    t.available_balance,

    c.card_status,

    m.risk_level AS merchant_risk,

    d.trusted_device,

    cu.risk_category AS customer_risk,

    k.risk_rating AS kyc_risk,
    k.pep_flag,
    k.sanctions_flag,

    (
        CASE
            WHEN t.transaction_amount >= 150000 THEN 25
            WHEN t.transaction_amount >= 100000 THEN 15
            WHEN t.transaction_amount >= 50000 THEN 10
            ELSE 0
        END

        +

        CASE
            WHEN m.risk_level = 'High' THEN 20
            WHEN m.risk_level = 'Medium' THEN 10
            ELSE 0
        END

        +

        CASE
            WHEN cu.risk_category = 'High' THEN 20
            WHEN cu.risk_category = 'Medium' THEN 10
            ELSE 0
        END

        +

        CASE
            WHEN k.risk_rating = 'High' THEN 20
            WHEN k.risk_rating = 'Medium' THEN 10
            ELSE 0
        END

        +

        CASE
            WHEN d.trusted_device = FALSE THEN 15
            ELSE 0
        END

        +

        CASE
            WHEN c.card_status = 'Blocked' THEN 25
            ELSE 0
        END

        +

        CASE
            WHEN k.pep_flag = TRUE THEN 20
            ELSE 0
        END

        +

        CASE
            WHEN k.sanctions_flag = TRUE THEN 40
            ELSE 0
        END

        +

        CASE
            WHEN t.transaction_status IN ('Failed', 'Reversed') THEN 10
            ELSE 0
        END

        +

        CASE
            WHEN t.transaction_type IN ('Cash Withdrawal', 'IMPS', 'RTGS') THEN 5
            ELSE 0
        END

    ) AS risk_score

FROM banking.transactions t

JOIN banking.accounts a
    ON a.account_id = t.account_id

JOIN banking.customers cu
    ON cu.customer_id = a.customer_id

LEFT JOIN banking.cards c
    ON c.card_id = t.card_id

LEFT JOIN banking.merchants m
    ON m.merchant_id = t.merchant_id

LEFT JOIN banking.devices d
    ON d.device_id = t.device_id

LEFT JOIN banking.customer_kyc k
    ON k.customer_id = cu.customer_id

ORDER BY t.transaction_id;
"""


# ---------------------------------------------------------------------------
# RISK LEVEL
# ---------------------------------------------------------------------------

def risk_level(score: int) -> str:

    if score >= 70:
        return "Critical"

    if score >= 50:
        return "High"

    if score >= 30:
        return "Medium"

    return "Low"


# ---------------------------------------------------------------------------
# ALERT EXPLANATION
# ---------------------------------------------------------------------------

def build_alert_reason(row) -> str:

    reasons = []

    amount = row["transaction_amount"]

    if amount >= 150000:
        reasons.append("Very high transaction amount")

    elif amount >= 100000:
        reasons.append("High transaction amount")

    elif amount >= 50000:
        reasons.append("Elevated transaction amount")


    if row["merchant_risk"] == "High":
        reasons.append("High-risk merchant")

    elif row["merchant_risk"] == "Medium":
        reasons.append("Medium-risk merchant")


    if row["customer_risk"] == "High":
        reasons.append("High-risk customer")

    elif row["customer_risk"] == "Medium":
        reasons.append("Medium-risk customer")


    if row["kyc_risk"] == "High":
        reasons.append("High KYC risk")

    elif row["kyc_risk"] == "Medium":
        reasons.append("Medium KYC risk")


    if row["trusted_device"] is False:
        reasons.append("Untrusted device")


    if row["card_status"] == "Blocked":
        reasons.append("Blocked card")


    if row["pep_flag"]:
        reasons.append("PEP flag")


    if row["sanctions_flag"]:
        reasons.append("Sanctions flag")


    if row["transaction_status"] in ("Failed", "Reversed"):
        reasons.append(
            f"Transaction status: {row['transaction_status']}"
        )


    if row["transaction_type"] in (
        "Cash Withdrawal",
        "IMPS",
        "RTGS"
    ):
        reasons.append(
            f"Risk-sensitive transaction type: {row['transaction_type']}"
        )


    if not reasons:
        reasons.append("Multiple transaction risk indicators")


    return "; ".join(reasons)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("EBRIP TRANSACTION FRAUD RULE ENGINE")
    print("=" * 70)

    print("\nLoading database configuration...")

    if not DB_CONFIG["password"]:
        print("ERROR: DB_PASSWORD is missing from .env")
        sys.exit(1)

    print(f"Database host : {DB_CONFIG['host']}")
    print(f"Database port : {DB_CONFIG['port']}")
    print(f"Database name : {DB_CONFIG['dbname']}")
    print(f"Database user : {DB_CONFIG['user']}")

    print("\nConnecting to PostgreSQL...")

    try:

        conn = psycopg2.connect(**DB_CONFIG)

    except Exception as exc:

        print(f"Database connection failed: {exc}")
        sys.exit(1)

    print("Database connection: OK")


    try:

        with conn.cursor() as cur:

            print("\nScoring transactions...")

            cur.execute(SCORE_SQL)

            rows = cur.fetchall()

            columns = [
                description[0]
                for description in cur.description
            ]

            print(f"Transactions evaluated: {len(rows):,}")


            alerts_created = {
                "Critical": 0,
                "High": 0,
            }

            skipped = 0


            for values in rows:

                row = dict(zip(columns, values))

                score = int(row["risk_score"])

                level = risk_level(score)


                # Only High/Critical transactions become alerts.
                if level not in ("Critical", "High"):
                    continue


                reason = build_alert_reason(row)


                cur.execute(
                    """
                    INSERT INTO risk.fraud_alerts
                    (
                        transaction_id,
                        detection_source,
                        fraud_score,
                        risk_level,
                        alert_status,
                        investigation_notes
                    )

                    SELECT
                        %s,
                        'Rule Engine',
                        %s,
                        %s,
                        'Open',
                        %s

                    WHERE NOT EXISTS
                    (
                        SELECT 1
                        FROM risk.fraud_alerts
                        WHERE transaction_id = %s
                          AND detection_source = 'Rule Engine'
                    );
                    """,

                    (
                        row["transaction_id"],
                        min(score / 100.0, 1.0),
                        level,
                        reason,
                        row["transaction_id"],
                    ),
                )


                if cur.rowcount == 1:

                    alerts_created[level] += 1

                else:

                    skipped += 1


            conn.commit()


            total_created = sum(
                alerts_created.values()
            )


            print("\n" + "=" * 70)
            print("ALERT CREATION SUMMARY")
            print("=" * 70)

            print(
                f"Critical alerts : {alerts_created['Critical']}"
            )

            print(
                f"High alerts     : {alerts_created['High']}"
            )

            print(
                f"Total created   : {total_created}"
            )

            print(
                f"Duplicates skip : {skipped}"
            )


    except Exception as exc:

        conn.rollback()

        print(
            f"\nRule engine failed: {exc}"
        )

        sys.exit(1)


    finally:

        conn.close()


    print("\nRule engine completed successfully.")


if __name__ == "__main__":
    main()

