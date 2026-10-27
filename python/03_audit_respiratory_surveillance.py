"""
03_audit_respiratory_surveillance.py

Summarise the manually audited Singapore respiratory
surveillance sources.

This script does NOT perform forecasting.

Its purpose is to determine which variables are suitable
for building the final weekly modelling dataset.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "surveillance_source_audit.csv"
)


def main():

    print("=" * 70)
    print("SINGAPORE RESPIRATORY SURVEILLANCE AUDIT")
    print("=" * 70)

    if not AUDIT_FILE.exists():
        raise FileNotFoundError(
            f"Audit file not found: {AUDIT_FILE}"
        )

    df = pd.read_csv(AUDIT_FILE)

    print("\nAudit file:")
    print(AUDIT_FILE)

    print("\nNumber of bulletin observations:")
    print(len(df))

    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")

    if len(df) == 0:
        print("\nNo bulletin observations have been entered yet.")
        return

    print("\nYears represented:")
    print(sorted(df["year"].dropna().unique()))

    print("\nURTI availability:")
    print(df["urti_available"].value_counts(dropna=False))

    print("\nInfluenza availability:")
    print(df["influenza_available"].value_counts(dropna=False))

    print("\nRSV availability:")
    print(df["rsv_available"].value_counts(dropna=False))

    print("\nCOVID availability:")
    print(df["covid_available"].value_counts(dropna=False))


if __name__ == "__main__":
    main()