"""
20_build_multipathogen_dataset.py

Build the weekly multi-pathogen analysis dataset by combining:

1. Weekly ARI polyclinic attendance
2. Influenza positivity among community ILI samples
3. COVID-19 positivity among community ARI samples

Primary multi-pathogen study period:
    2024 Week 09 through 2026 Week 41

This script only performs the clean merge and QC.
Lagged pathogen predictors will be created in a later step.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARI_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ari_weekly_clean.csv"
)

PATHOGEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_pathogen_signals.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multipathogen_weekly_dataset.csv"
)

QC_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_merge_qc.csv"
)


# ============================================================
# STUDY PERIOD
# ============================================================

START_YEAR = 2024
START_WEEK = 9

END_YEAR = 2026
END_WEEK = 41


# ============================================================
# HELPERS
# ============================================================

def within_study_period(df):
    """
    Restrict data to:
        2024 Week 09 -> 2026 Week 41
    """

    keep = (
        (
            (df["year"] > START_YEAR)
            |
            (
                (df["year"] == START_YEAR)
                &
                (df["epi_week"] >= START_WEEK)
            )
        )
        &
        (
            (df["year"] < END_YEAR)
            |
            (
                (df["year"] == END_YEAR)
                &
                (df["epi_week"] <= END_WEEK)
            )
        )
    )

    return (
        df.loc[keep]
        .copy()
        .reset_index(drop=True)
    )


def identify_ari_column(df):
    """
    Identify the ARI attendance column.

    Earlier stages of the project normally use
    'ari_attendance', but a few reasonable alternatives
    are supported so the script fails gracefully.
    """

    candidates = [
        "ari_attendance",
        "average_daily_ari_attendance",
        "ari_polyclinic_attendance",
        "weekly_ari_attendance",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify the ARI attendance column.\n"
        f"Available columns are:\n{list(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("BUILD MULTI-PATHOGEN WEEKLY DATASET")
    print("=" * 72)

    # --------------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------------

    if not ARI_FILE.exists():

        raise FileNotFoundError(
            f"ARI file not found:\n{ARI_FILE}"
        )

    if not PATHOGEN_FILE.exists():

        raise FileNotFoundError(
            f"Pathogen file not found:\n{PATHOGEN_FILE}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    ari = pd.read_csv(ARI_FILE)

    pathogen = pd.read_csv(
        PATHOGEN_FILE
    )

    print("\nARI input:")
    print(ARI_FILE)
    print("Rows:", len(ari))

    print("\nPathogen input:")
    print(PATHOGEN_FILE)
    print("Rows:", len(pathogen))

    # --------------------------------------------------------
    # REQUIRED KEY COLUMNS
    # --------------------------------------------------------

    required_keys = [
        "year",
        "epi_week",
    ]

    for column in required_keys:

        if column not in ari.columns:

            raise ValueError(
                f"ARI dataset is missing "
                f"required column: {column}"
            )

        if column not in pathogen.columns:

            raise ValueError(
                f"Pathogen dataset is missing "
                f"required column: {column}"
            )

    # --------------------------------------------------------
    # IDENTIFY ARI OUTCOME
    # --------------------------------------------------------

    ari_column = identify_ari_column(
        ari
    )

    print(
        f"\nARI outcome column detected: "
        f"{ari_column}"
    )

    if ari_column != "ari_attendance":

        ari = ari.rename(
            columns={
                ari_column:
                    "ari_attendance"
            }
        )

        ari_column = "ari_attendance"

    # --------------------------------------------------------
    # STANDARDIZE KEY TYPES
    # --------------------------------------------------------

    for df in [ari, pathogen]:

        df["year"] = pd.to_numeric(
            df["year"],
            errors="raise",
        ).astype(int)

        df["epi_week"] = pd.to_numeric(
            df["epi_week"],
            errors="raise",
        ).astype(int)

    # --------------------------------------------------------
    # RESTRICT TO MULTI-PATHOGEN PERIOD
    # --------------------------------------------------------

    ari_period = within_study_period(
        ari
    )

    pathogen_period = (
        within_study_period(
            pathogen
        )
    )

    print(
        "\nARI rows in study period:",
        len(ari_period),
    )

    print(
        "Pathogen rows in study period:",
        len(pathogen_period),
    )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    ari_duplicates = int(
        ari_period.duplicated(
            subset=[
                "year",
                "epi_week",
            ]
        ).sum()
    )

    pathogen_duplicates = int(
        pathogen_period.duplicated(
            subset=[
                "year",
                "epi_week",
            ]
        ).sum()
    )

    print(
        "\nARI duplicate year/week rows:",
        ari_duplicates,
    )

    print(
        "Pathogen duplicate year/week rows:",
        pathogen_duplicates,
    )

    if ari_duplicates != 0:

        raise RuntimeError(
            "ARI dataset contains duplicate "
            "year/week rows."
        )

    if pathogen_duplicates != 0:

        raise RuntimeError(
            "Pathogen dataset contains duplicate "
            "year/week rows."
        )

    # --------------------------------------------------------
    # KEEP PATHOGEN VARIABLES REQUIRED FOR MODELLING
    # --------------------------------------------------------

    pathogen_columns = [
        "year",
        "epi_week",
        "influenza_positivity",
        "influenza_censored",
        "influenza_reported_threshold",
        "influenza_reported_text",
        "covid_positivity",
        "covid_censored",
        "covid_reported_threshold",
        "covid_reported_text",
    ]

    missing_pathogen_columns = [
        column
        for column in pathogen_columns
        if column not in pathogen_period.columns
    ]

    if missing_pathogen_columns:

        raise ValueError(
            "Missing expected pathogen columns:\n"
            f"{missing_pathogen_columns}"
        )

    pathogen_period = (
        pathogen_period[
            pathogen_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    merged = ari_period.merge(
        pathogen_period,
        on=[
            "year",
            "epi_week",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    # --------------------------------------------------------
    # MERGE QC
    # --------------------------------------------------------

    merge_summary = (
        merged["_merge"]
        .value_counts()
        .rename_axis("merge_status")
        .reset_index(name="n")
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "MERGE SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        merge_summary.to_string(
            index=False
        )
    )

    unmatched = merged[
        merged["_merge"]
        != "both"
    ].copy()

    if not unmatched.empty:

        print(
            "\nWARNING:"
            " Some ARI weeks did not match "
            "the pathogen dataset."
        )

        print(
            unmatched[
                [
                    "year",
                    "epi_week",
                    "_merge",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # REMOVE MERGE INDICATOR
    # --------------------------------------------------------

    merged = merged.drop(
        columns="_merge"
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    merged = (
        merged
        .sort_values(
            [
                "year",
                "epi_week",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # FINAL QC
    # --------------------------------------------------------

    duplicate_final = int(
        merged.duplicated(
            subset=[
                "year",
                "epi_week",
            ]
        ).sum()
    )

    missing_ari = int(
        merged[
            "ari_attendance"
        ]
        .isna()
        .sum()
    )

    missing_influenza = int(
        merged[
            "influenza_positivity"
        ]
        .isna()
        .sum()
    )

    missing_covid = int(
        merged[
            "covid_positivity"
        ]
        .isna()
        .sum()
    )

    expected_rows = (
        44
        + 53
        + 41
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL DATASET QC"
    )

    print(
        "=" * 72
    )

    print(
        f"Expected rows: "
        f"{expected_rows}"
    )

    print(
        f"Observed rows: "
        f"{len(merged)}"
    )

    print(
        f"Duplicate year/week rows: "
        f"{duplicate_final}"
    )

    print(
        f"Missing ARI values: "
        f"{missing_ari}"
    )

    print(
        f"Missing influenza values: "
        f"{missing_influenza}"
    )

    print(
        f"Missing COVID values: "
        f"{missing_covid}"
    )

    # --------------------------------------------------------
    # YEARLY SUMMARY
    # --------------------------------------------------------

    yearly_summary = (
        merged
        .groupby("year")
        .agg(
            weeks=(
                "epi_week",
                "size",
            ),

            ari_available=(
                "ari_attendance",
                "count",
            ),

            influenza_available=(
                "influenza_positivity",
                "count",
            ),

            covid_available=(
                "covid_positivity",
                "count",
            ),
        )
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "OBSERVATIONS BY YEAR"
    )

    print(
        "=" * 72
    )

    print(
        yearly_summary.to_string()
    )

    # --------------------------------------------------------
    # IMPORTANT VALIDATION WEEKS
    # --------------------------------------------------------

    recent = merged[
        (merged["year"] == 2026)
        &
        (
            merged["epi_week"]
            .isin(
                [
                    39,
                    40,
                    41,
                ]
            )
        )
    ]

    print(
        "\n" + "=" * 72
    )

    print(
        "2026 WEEKS 39-41 CHECK"
    )

    print(
        "=" * 72
    )

    print(
        recent[
            [
                "year",
                "epi_week",
                "ari_attendance",
                "influenza_positivity",
                "covid_positivity",
                "covid_censored",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    merge_summary.to_csv(
        QC_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 72
    )

    print(
        f"\nMerged dataset:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nMerge QC:\n"
        f"{QC_FILE}"
    )


if __name__ == "__main__":
    main()