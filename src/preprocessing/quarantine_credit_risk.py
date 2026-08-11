from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "credit_risk"
    / "cs-training.csv"
)

QUARANTINE_DIR = (
    PROJECT_ROOT
    / "data"
    / "quarantine"
    / "credit_risk"
)

STAGING_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
)

QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    invalid_age_mask = (
        (df["age"] < 18)
        | (df["age"] > 100)
    )

    invalid_rows = df.loc[invalid_age_mask].copy()
    valid_rows = df.loc[~invalid_age_mask].copy()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Add audit metadata to quarantined records.
    invalid_rows["quarantine_rule"] = "age_range"
    invalid_rows["quarantine_reason"] = "Age outside allowed range 18-100"
    invalid_rows["quarantined_at_utc"] = timestamp
    invalid_rows["source_file"] = RAW_FILE.name

    quarantine_file = (
        QUARANTINE_DIR
        / "credit_risk_invalid_age.csv"
    )

    staging_file = (
        STAGING_DIR
        / "credit_risk_clean.csv"
    )

    invalid_rows.to_csv(
        quarantine_file,
        index=False,
    )

    valid_rows.to_csv(
        staging_file,
        index=False,
    )

    audit = {
        "dataset": RAW_FILE.name,
        "processed_at_utc": timestamp,
        "source_rows": int(len(df)),
        "valid_rows": int(len(valid_rows)),
        "quarantined_rows": int(len(invalid_rows)),
        "quarantine_rule": "age_range",
        "age_min": 18,
        "age_max": 100,
        "quarantine_file": str(quarantine_file),
        "clean_staging_file": str(staging_file),
    }

    audit_file = (
        QUARANTINE_DIR
        / "quarantine_audit.json"
    )

    with audit_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=4,
        )

    print("=" * 70)
    print("CREDIT RISK QUARANTINE PIPELINE")
    print("=" * 70)
    print(f"Source rows       : {len(df):,}")
    print(f"Valid rows        : {len(valid_rows):,}")
    print(f"Quarantined rows  : {len(invalid_rows):,}")
    print()
    print(f"Quarantine file   : {quarantine_file}")
    print(f"Clean staging     : {staging_file}")
    print(f"Audit file        : {audit_file}")


if __name__ == "__main__":
    main()
