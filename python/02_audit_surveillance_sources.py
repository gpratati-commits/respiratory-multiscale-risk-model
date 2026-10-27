"""
02_audit_surveillance_sources.py

Purpose
-------
Audit the availability and consistency of Singapore respiratory
surveillance data before constructing the modelling dataset.

Primary candidate outcome:
    Weekly average daily polyclinic attendance for URTI.

Questions:
1. What years are available?
2. Is the outcome definition consistent?
3. Are epidemiological weeks complete?
4. What influenza variables are available?
5. Are RSV and SARS-CoV-2 time series consistently available?
6. Were there surveillance-definition changes?
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "singapore_surveillance"


def main():
    print("Singapore respiratory surveillance source audit")
    print("=" * 60)
    print(f"Raw data directory: {RAW_DIR}")

    if RAW_DIR.exists():
        print("STATUS: surveillance directory exists.")
    else:
        print("ERROR: surveillance directory is missing.")


if __name__ == "__main__":
    main()