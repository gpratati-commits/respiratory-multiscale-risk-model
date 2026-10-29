"""
08_assess_ari_data_quality.py

Quality-control assessment of the extracted Singapore weekly
Acute Respiratory Infection (ARI) polyclinic attendance series.

Inputs
------
data/interim/surveillance_source_index.csv
data/interim/ari_weekly_extraction.csv

Outputs
-------
data/processed/ari_weekly_clean.csv
outputs/tables/ari_data_quality_summary.csv
outputs/tables/ari_flagged_observations.csv

Important:
This script flags unusual observations for review.
It does NOT automatically delete outliers.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_INDEX = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "surveillance_source_index.csv"
)

ARI_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ari_weekly_extraction.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

CLEAN_OUTPUT = (
    PROCESSED_DIR
    / "ari_weekly_clean.csv"
)

SUMMARY_OUTPUT = (
    TABLE_DIR
    / "ari_data_quality_summary.csv"
)

FLAGGED_OUTPUT = (
    TABLE_DIR
    / "ari_flagged_observations.csv"
)


def main():

    print("=" * 70)
    print("ARI WEEKLY DATA QUALITY ASSESSMENT")
    print("=" * 70)

    if not SOURCE_INDEX.exists():
        raise FileNotFoundError(
            f"Source index not found:\n{SOURCE_INDEX}"
        )

    if not ARI_FILE.exists():
        raise FileNotFoundError(
            f"ARI extraction file not found:\n{ARI_FILE}"
        )

    source = pd.read_csv(SOURCE_INDEX)

    ari = pd.read_csv(ARI_FILE)

    required_columns = {
        "year",
        "epi_week",
        "ari_attendance",
    }

    missing_columns = (
        required_columns
        - set(ari.columns)
    )

    if missing_columns:
        raise ValueError(
            "ARI file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 1. Check duplicate year/week observations
    # ---------------------------------------------------------

    duplicate_mask = ari.duplicated(
        subset=["year", "epi_week"],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    # ---------------------------------------------------------
    # 2. Compare extraction with complete source index
    # ---------------------------------------------------------

    expected = (
        source[
            ["year", "epi_week"]
        ]
        .drop_duplicates()
        .sort_values(
            ["year", "epi_week"]
        )
        .reset_index(drop=True)
    )

    merged = expected.merge(
        ari,
        on=["year", "epi_week"],
        how="left",
        validate="one_to_one",
    )

    merged["ari_attendance"] = pd.to_numeric(
        merged["ari_attendance"],
        errors="coerce",
    )

    missing_ari = int(
        merged["ari_attendance"]
        .isna()
        .sum()
    )

    # ---------------------------------------------------------
    # 3. Basic validity checks
    # ---------------------------------------------------------

    merged["flag_nonpositive"] = (
        merged["ari_attendance"] <= 0
    )

    nonpositive_count = int(
        merged["flag_nonpositive"]
        .fillna(False)
        .sum()
    )

    # ---------------------------------------------------------
    # 4. Overall IQR outlier flag
    #
    # This identifies unusual observations but DOES NOT delete them.
    # ---------------------------------------------------------

    valid_values = (
        merged["ari_attendance"]
        .dropna()
    )

    q1 = valid_values.quantile(0.25)
    q3 = valid_values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    merged["flag_iqr"] = (
        (merged["ari_attendance"] < lower_bound)
        |
        (merged["ari_attendance"] > upper_bound)
    )

    # ---------------------------------------------------------
    # 5. Sequential weekly-change flag
    #
    # Large week-to-week changes are flagged for manual review.
    # The threshold is based on the distribution of observed changes,
    # rather than an arbitrary fixed percentage.
    # ---------------------------------------------------------

    merged = merged.sort_values(
        ["year", "epi_week"]
    ).reset_index(drop=True)

    merged["previous_ari"] = (
        merged["ari_attendance"]
        .shift(1)
    )

    merged["absolute_weekly_change"] = (
        merged["ari_attendance"]
        - merged["previous_ari"]
    ).abs()

    weekly_changes = (
        merged["absolute_weekly_change"]
        .dropna()
    )

    change_q1 = weekly_changes.quantile(0.25)
    change_q3 = weekly_changes.quantile(0.75)

    change_iqr = (
        change_q3
        - change_q1
    )

    change_upper = (
        change_q3
        + 3.0 * change_iqr
    )

    merged["flag_large_weekly_change"] = (
        merged["absolute_weekly_change"]
        > change_upper
    )

    # ---------------------------------------------------------
    # 6. Create review flag
    # ---------------------------------------------------------

    merged["needs_review"] = (
        merged["ari_attendance"].isna()
        |
        merged["flag_nonpositive"].fillna(False)
        |
        merged["flag_iqr"].fillna(False)
        |
        merged[
            "flag_large_weekly_change"
        ].fillna(False)
    )

    # ---------------------------------------------------------
    # 7. Add simple sequential observation number
    #
    # We deliberately do NOT convert epidemiological week to
    # ISO-calendar dates here. Singapore bulletin E-weeks should
    # not be assumed to be identical to ISO weeks without verifying
    # the official epidemiological-week convention.
    # ---------------------------------------------------------

    merged["time_index"] = np.arange(
        1,
        len(merged) + 1,
    )

    # ---------------------------------------------------------
    # 8. Summary table
    # ---------------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "expected_bulletins",
                "extracted_rows",
                "duplicate_rows",
                "missing_ari_values",
                "nonpositive_values",
                "iqr_flagged_values",
                "large_weekly_change_flags",
                "total_review_flags",
                "minimum_ari",
                "maximum_ari",
                "median_ari",
                "mean_ari",
                "iqr_lower_bound",
                "iqr_upper_bound",
            ],
            "value": [
                len(expected),
                len(ari),
                duplicate_count,
                missing_ari,
                nonpositive_count,
                int(
                    merged["flag_iqr"]
                    .fillna(False)
                    .sum()
                ),
                int(
                    merged[
                        "flag_large_weekly_change"
                    ]
                    .fillna(False)
                    .sum()
                ),
                int(
                    merged["needs_review"]
                    .fillna(False)
                    .sum()
                ),
                valid_values.min(),
                valid_values.max(),
                valid_values.median(),
                valid_values.mean(),
                lower_bound,
                upper_bound,
            ],
        }
    )

    # ---------------------------------------------------------
    # 9. Save outputs
    # ---------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_columns = [
        "year",
        "epi_week",
        "time_index",
        "ari_attendance",
    ]

    clean_df = merged[
        clean_columns
    ].copy()

    clean_df.to_csv(
        CLEAN_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    flagged = merged[
        merged["needs_review"]
    ].copy()

    flagged.to_csv(
        FLAGGED_OUTPUT,
        index=False,
    )

    # ---------------------------------------------------------
    # 10. Print results
    # ---------------------------------------------------------

    print(
        f"\nExpected bulletin observations: "
        f"{len(expected)}"
    )

    print(
        f"Extracted observations: "
        f"{len(ari)}"
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_count}"
    )

    print(
        f"Missing ARI values: "
        f"{missing_ari}"
    )

    print(
        f"Non-positive ARI values: "
        f"{nonpositive_count}"
    )

    print(
        "\nARI descriptive statistics:"
    )

    print(
        valid_values.describe()
    )

    print(
        "\nIQR review bounds:"
    )

    print(
        f"Lower: {lower_bound:.2f}"
    )

    print(
        f"Upper: {upper_bound:.2f}"
    )

    print(
        f"\nObservations flagged for review: "
        f"{len(flagged)}"
    )

    print(
        f"\nClean weekly dataset saved to:\n"
        f"{CLEAN_OUTPUT}"
    )

    print(
        f"\nQuality summary saved to:\n"
        f"{SUMMARY_OUTPUT}"
    )

    print(
        f"\nFlagged observations saved to:\n"
        f"{FLAGGED_OUTPUT}"
    )


if __name__ == "__main__":
    main()