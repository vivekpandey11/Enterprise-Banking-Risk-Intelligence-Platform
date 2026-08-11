from pathlib import Path
import hashlib
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "credit_risk"
    / "cs-training.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "credit_risk"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def profile_dataset(file_path: Path) -> dict:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    profile = {
        "file": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "sha256": calculate_sha256(file_path),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_bytes": int(
            df.memory_usage(deep=True).sum()
        ),
    }

    for column in df.columns:

        series = df[column]

        profile["columns"][column] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_percentage": round(
                float(series.isna().mean() * 100),
                4,
            ),
            "unique_count": int(series.nunique(dropna=True)),
        }

    return profile


def main():

    print("=" * 70)
    print("ENTERPRISE BANKING RISK INTELLIGENCE PLATFORM")
    print("CREDIT RISK DATASET PROFILER")
    print("=" * 70)

    print(f"\nDataset: {DATA_FILE}")

    profile = profile_dataset(DATA_FILE)

    output_file = OUTPUT_DIR / "credit_risk_profile.json"

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            profile,
            file,
            indent=4,
        )

    print("\nDATASET SUMMARY")
    print("-" * 70)

    print(
        f"Rows           : {profile['row_count']:,}"
    )

    print(
        f"Columns        : {profile['column_count']}"
    )

    print(
        f"Duplicate rows : {profile['duplicate_rows']:,}"
    )

    print(
        f"File size      : "
        f"{profile['file_size_bytes']:,} bytes"
    )

    print(
        f"SHA256         : "
        f"{profile['sha256']}"
    )

    print("\nCOLUMN PROFILE")
    print("-" * 70)

    for column, metadata in profile["columns"].items():

        print(
            f"{column:35} "
            f"{metadata['dtype']:10} "
            f"NULL={metadata['null_count']:,} "
            f"({metadata['null_percentage']:.2f}%) "
            f"UNIQUE={metadata['unique_count']:,}"
        )

    print("\nProfile written to:")

    print(output_file)


if __name__ == "__main__":
    main()