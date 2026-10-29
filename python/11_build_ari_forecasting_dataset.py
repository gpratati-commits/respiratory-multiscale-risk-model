"""
11_build_ari_forecasting_dataset.py

Construct leakage-safe ARI forecasting datasets for
1-, 2-, and 4-week-ahead prediction.

Forecast origin:
    week t

Available ARI information at the forecast origin:
    ARI(t), ARI(t-1), ..., ARI(t-4)

Forecast targets:
    ARI(t+1)
    ARI(t+2)
    ARI(t+4)

Primary chronological design:
    Training targets:   2023-2024
    Validation targets: 2025
    Test targets:       2026

Important:
The split is assigned according to the TARGET week,
not according to the forecast-origin week.

Seasonality is represented using Fourier sine/cosine terms.

Output:
    data/processed/ari_forecasting_dataset.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ari_weekly_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ari_forecasting_dataset.csv"
)


# ============================================================
# FORECAST SETTINGS
# ============================================================

HORIZONS = [1, 2, 4]


# ============================================================
# SPLIT DEFINITION
# ============================================================

def assign_split(target_year):
    """
    Assign chronological train / validation / test split
    according to the TARGET year.

    Train:
        2023-2024

    Validation:
        2025

    Test:
        2026
    """

    if target_year in [2023, 2024]:
        return "train"

    if target_year == 2025:
        return "validation"

    if target_year == 2026:
        return "test"

    return "outside_analysis"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILD LEAKAGE-SAFE ARI FORECASTING DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # 2. Read cleaned ARI dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "year",
        "epi_week",
        "time_index",
        "ari_attendance",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Input ARI dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # 3. Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "time_index"
        )
        .reset_index(
            drop=True
        )
    )

    # Make sure ARI is numeric.
    df["ari_attendance"] = pd.to_numeric(
        df["ari_attendance"],
        errors="raise",
    )

    # --------------------------------------------------------
    # 4. Create forecast-origin ARI predictors
    # --------------------------------------------------------

    # Current observed ARI at forecast origin.
    df["ari_current"] = (
        df["ari_attendance"]
    )

    # Previous 1-4 observed weeks.
    for lag in range(1, 5):

        df[f"ari_lag_{lag}"] = (
            df["ari_attendance"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # 5. Create lookup table by epidemiological year/week
    #
    # This is safer than blindly shifting by 52 rows because
    # some epidemiological years have Week 53.
    # --------------------------------------------------------

    same_week_lookup = {
        (
            int(row.year),
            int(row.epi_week),
        ):
        float(row.ari_attendance)

        for row in df.itertuples()
    }

    # Same epidemiological week as the FORECAST ORIGIN
    # in the previous year.
    #
    # This is retained as a candidate predictor.
    df[
        "ari_origin_same_week_previous_year"
    ] = [

        same_week_lookup.get(
            (
                int(year) - 1,
                int(week),
            ),
            np.nan,
        )

        for year, week in zip(
            df["year"],
            df["epi_week"],
        )
    ]

    # --------------------------------------------------------
    # 6. Build records for every forecast horizon
    # --------------------------------------------------------

    records = []

    for horizon in HORIZONS:

        # Future ARI target.
        target_ari = (
            df["ari_attendance"]
            .shift(-horizon)
        )

        # Target year.
        target_year = (
            df["year"]
            .shift(-horizon)
        )

        # Target epidemiological week.
        target_week = (
            df["epi_week"]
            .shift(-horizon)
        )

        # Target sequential index.
        target_time_index = (
            df["time_index"]
            .shift(-horizon)
        )

        for i in range(
            len(df)
        ):

            # If target does not exist because we are at
            # the end of the observed series, skip it.
            if pd.isna(
                target_ari.iloc[i]
            ):
                continue

            target_year_i = int(
                target_year.iloc[i]
            )

            target_week_i = int(
                target_week.iloc[i]
            )

            target_time_index_i = int(
                target_time_index.iloc[i]
            )

            # ------------------------------------------------
            # Assign chronological split using TARGET year.
            # ------------------------------------------------

            split = assign_split(
                target_year_i
            )

            if split == "outside_analysis":
                continue

            # ------------------------------------------------
            # Correct seasonal-naive reference:
            #
            # Same epidemiological week as the TARGET,
            # but one year earlier.
            #
            # Example:
            # Forecast target = 2026 Week 14
            # Seasonal naive = 2025 Week 14
            # ------------------------------------------------

            target_same_week_previous_year = (
                same_week_lookup.get(
                    (
                        target_year_i - 1,
                        target_week_i,
                    ),
                    np.nan,
                )
            )

            # ------------------------------------------------
            # Fourier seasonality
            #
            # Approximate mean epidemiological-year length.
            # ------------------------------------------------

            seasonal_angle = (
                2.0
                * np.pi
                * (
                    target_week_i - 1
                )
                / 52.1775
            )

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            record = {

                # Forecast-origin information
                "origin_year":
                    int(
                        df.loc[
                            i,
                            "year",
                        ]
                    ),

                "origin_epi_week":
                    int(
                        df.loc[
                            i,
                            "epi_week",
                        ]
                    ),

                "origin_time_index":
                    int(
                        df.loc[
                            i,
                            "time_index",
                        ]
                    ),

                # Forecast horizon
                "horizon_weeks":
                    int(
                        horizon
                    ),

                # Target information
                "target_year":
                    target_year_i,

                "target_epi_week":
                    target_week_i,

                "target_time_index":
                    target_time_index_i,

                "target_ari":
                    float(
                        target_ari.iloc[i]
                    ),

                # Chronological split
                "split":
                    split,

                # Current and lagged ARI predictors
                "ari_current":
                    float(
                        df.loc[
                            i,
                            "ari_current",
                        ]
                    ),

                "ari_lag_1":
                    df.loc[
                        i,
                        "ari_lag_1",
                    ],

                "ari_lag_2":
                    df.loc[
                        i,
                        "ari_lag_2",
                    ],

                "ari_lag_3":
                    df.loc[
                        i,
                        "ari_lag_3",
                    ],

                "ari_lag_4":
                    df.loc[
                        i,
                        "ari_lag_4",
                    ],

                # Same week as forecast origin,
                # previous year.
                "ari_origin_same_week_previous_year":
                    df.loc[
                        i,
                        "ari_origin_same_week_previous_year",
                    ],

                # Same week as TARGET,
                # previous year.
                #
                # This is the correct seasonal-naive
                # benchmark predictor.
                "ari_target_same_week_previous_year":
                    target_same_week_previous_year,

                # First Fourier harmonic
                "season_sin_1":
                    np.sin(
                        seasonal_angle
                    ),

                "season_cos_1":
                    np.cos(
                        seasonal_angle
                    ),

                # Second Fourier harmonic
                "season_sin_2":
                    np.sin(
                        2
                        * seasonal_angle
                    ),

                "season_cos_2":
                    np.cos(
                        2
                        * seasonal_angle
                    ),
            }

            records.append(
                record
            )

    # --------------------------------------------------------
    # 7. Convert to DataFrame
    # --------------------------------------------------------

    forecast = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # 8. Require complete recent ARI history
    #
    # We require current ARI plus lags 1-4.
    #
    # We DO NOT require previous-year values because Week 53
    # may legitimately have no matching value in some years.
    # --------------------------------------------------------

    required_short_lag_features = [
        "ari_current",
        "ari_lag_1",
        "ari_lag_2",
        "ari_lag_3",
        "ari_lag_4",
    ]

    before_drop = len(
        forecast
    )

    forecast = (
        forecast
        .dropna(
            subset=
            required_short_lag_features
        )
        .reset_index(
            drop=True
        )
    )

    removed_rows = (
        before_drop
        - len(forecast)
    )

    # --------------------------------------------------------
    # 9. Sort final forecasting dataset
    # --------------------------------------------------------

    forecast = (
        forecast
        .sort_values(
            [
                "horizon_weeks",
                "target_time_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 10. Save output
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 11. Print summary
    # --------------------------------------------------------

    print(
        f"\nRows removed because "
        f"short-lag history was unavailable: "
        f"{removed_rows}"
    )

    print(
        f"\nFinal forecasting rows: "
        f"{len(forecast)}"
    )

    print(
        "\nROWS BY HORIZON AND SPLIT"
    )

    split_summary = (
        forecast
        .groupby(
            [
                "horizon_weeks",
                "split",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        split_summary
    )

    # --------------------------------------------------------
    # 12. Target-year check
    # --------------------------------------------------------

    print(
        "\nTARGET YEARS BY SPLIT"
    )

    target_year_summary = (
        forecast
        .groupby(
            [
                "split",
                "target_year",
            ]
        )
        .size()
    )

    print(
        target_year_summary
    )

    # --------------------------------------------------------
    # 13. Missing seasonal-naive values
    # --------------------------------------------------------

    print(
        "\nMISSING TARGET-WEEK "
        "PREVIOUS-YEAR VALUES"
    )

    seasonal_missing = (
        forecast
        .groupby(
            [
                "split",
                "horizon_weeks",
            ]
        )[
            "ari_target_same_week_previous_year"
        ]
        .apply(
            lambda x:
            x.isna().sum()
        )
    )

    print(
        seasonal_missing
    )

    # --------------------------------------------------------
    # 14. Final output location
    # --------------------------------------------------------

    print(
        "\nForecasting dataset saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()