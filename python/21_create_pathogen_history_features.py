"""
21_create_pathogen_history_features.py

Create leakage-safe influenza and COVID-19 history features
for multi-pathogen ARI forecasting.

Input
-----
data/processed/multipathogen_weekly_dataset.csv

Output
------
data/processed/multipathogen_feature_dataset.csv

outputs/tables/multipathogen_feature_qc.csv

Important
---------
All pathogen forecasting features use information from prior weeks.

No future pathogen observations are used.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multipathogen_weekly_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multipathogen_feature_dataset.csv"
)

QC_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_feature_qc.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("CREATE LEAKAGE-SAFE PATHOGEN HISTORY FEATURES")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nInput rows: {len(df)}"
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "year",
        "epi_week",
        "ari_attendance",
        "influenza_positivity",
        "covid_positivity",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "year",
                "epi_week",
            ]
        )
        .reset_index(drop=True)
    )

    # Sequential week counter.
    df["time_index"] = range(
        1,
        len(df) + 1,
    )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    duplicates = int(
        df.duplicated(
            subset=[
                "year",
                "epi_week",
            ]
        ).sum()
    )

    if duplicates != 0:

        raise RuntimeError(
            "Duplicate year/week rows detected."
        )

    # --------------------------------------------------------
    # CREATE LAG FEATURES
    # --------------------------------------------------------

    for lag in [1, 2, 4]:

        df[
            f"influenza_lag{lag}"
        ] = (
            df[
                "influenza_positivity"
            ]
            .shift(lag)
        )

        df[
            f"covid_lag{lag}"
        ] = (
            df[
                "covid_positivity"
            ]
            .shift(lag)
        )

    # --------------------------------------------------------
    # RECENT HISTORY MEANS
    #
    # shift(1) is essential:
    # current/future pathogen activity is not included.
    # --------------------------------------------------------

    influenza_previous = (
        df[
            "influenza_positivity"
        ]
        .shift(1)
    )

    covid_previous = (
        df[
            "covid_positivity"
        ]
        .shift(1)
    )

    df[
        "influenza_recent_2week_mean"
    ] = (
        influenza_previous
        .rolling(
            window=2,
            min_periods=2,
        )
        .mean()
    )

    df[
        "influenza_recent_4week_mean"
    ] = (
        influenza_previous
        .rolling(
            window=4,
            min_periods=4,
        )
        .mean()
    )

    df[
        "covid_recent_2week_mean"
    ] = (
        covid_previous
        .rolling(
            window=2,
            min_periods=2,
        )
        .mean()
    )

    df[
        "covid_recent_4week_mean"
    ] = (
        covid_previous
        .rolling(
            window=4,
            min_periods=4,
        )
        .mean()
    )

    # --------------------------------------------------------
    # OPTIONAL RECENT CHANGE FEATURES
    # --------------------------------------------------------

    df[
        "influenza_weekly_change"
    ] = (
        df[
            "influenza_lag1"
        ]
        -
        df[
            "influenza_lag2"
        ]
    )

    df[
        "covid_weekly_change"
    ] = (
        df[
            "covid_lag1"
        ]
        -
        df[
            "covid_lag2"
        ]
    )

    # --------------------------------------------------------
    # FLAG ROWS WITH COMPLETE PATHOGEN HISTORY
    # --------------------------------------------------------

    model_features = [
        "influenza_lag1",
        "influenza_lag2",
        "influenza_lag4",
        "influenza_recent_2week_mean",
        "influenza_recent_4week_mean",
        "covid_lag1",
        "covid_lag2",
        "covid_lag4",
        "covid_recent_2week_mean",
        "covid_recent_4week_mean",
    ]

    df[
        "complete_pathogen_history"
    ] = (
        df[
            model_features
        ]
        .notna()
        .all(axis=1)
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print("FEATURE QC")

    print("=" * 72)

    print(
        "Duplicate year/week rows:",
        duplicates,
    )

    print(
        "Rows with complete pathogen history:",
        int(
            df[
                "complete_pathogen_history"
            ].sum()
        ),
    )

    print(
        "Rows without complete pathogen history:",
        int(
            (
                ~df[
                    "complete_pathogen_history"
                ]
            ).sum()
        ),
    )

    print(
        "\nMissing values by pathogen feature:"
    )

    missing_summary = (
        df[
            model_features
        ]
        .isna()
        .sum()
    )

    print(
        missing_summary.to_string()
    )

    # --------------------------------------------------------
    # VALIDATION EXAMPLE
    # --------------------------------------------------------

    check = df[
        (df["year"] == 2026)
        &
        (
            df["epi_week"]
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

    print("2026 WEEKS 39-41 FEATURE CHECK")

    print("=" * 72)

    columns_to_show = [
        "year",
        "epi_week",
        "influenza_positivity",
        "influenza_lag1",
        "influenza_lag2",
        "influenza_lag4",
        "covid_positivity",
        "covid_lag1",
        "covid_lag2",
        "covid_lag4",
    ]

    print(
        check[
            columns_to_show
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # QC TABLE
    # --------------------------------------------------------

    qc = pd.DataFrame(
        {
            "feature":
                model_features,

            "missing_count":
                [
                    int(
                        df[column]
                        .isna()
                        .sum()
                    )
                    for column
                    in model_features
                ],
        }
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    qc.to_csv(
        QC_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 72
    )

    print("OUTPUT FILES")

    print("=" * 72)

    print(
        f"\nFeature dataset:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nQC table:\n"
        f"{QC_FILE}"
    )


if __name__ == "__main__":
    main()