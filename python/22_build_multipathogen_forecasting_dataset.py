"""
22_build_multipathogen_forecasting_dataset.py

Construct leakage-safe 1-, 2-, and 4-week-ahead forecasting
observations for the multi-pathogen ARI analysis.

Input:
    data/processed/multipathogen_feature_dataset.csv

Output:
    data/processed/multipathogen_forecasting_dataset.csv

Splits are determined from TARGET YEAR:
    2024 -> train
    2025 -> validation
    2026 -> test
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
    / "multipathogen_feature_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multipathogen_forecasting_dataset.csv"
)

QC_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_forecasting_dataset_qc.csv"
)


# ============================================================
# SETTINGS
# ============================================================

HORIZONS = [1, 2, 4]


def assign_split(target_year):
    """Assign chronological split using the target year."""

    if target_year == 2024:
        return "train"

    if target_year == 2025:
        return "validation"

    if target_year == 2026:
        return "test"

    return None


def main():

    print("=" * 72)
    print("BUILD MULTI-PATHOGEN FORECASTING DATASET")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df = (
        df
        .sort_values(["year", "epi_week"])
        .reset_index(drop=True)
    )

    required = [
        "year",
        "epi_week",
        "ari_attendance",
        "complete_pathogen_history",
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

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns:\n{missing}"
        )

    # --------------------------------------------------------
    # ARI HISTORY FEATURES
    # --------------------------------------------------------

    df["ari_current"] = df["ari_attendance"]

    df["ari_lag1"] = (
        df["ari_attendance"]
        .shift(1)
    )

    df["ari_lag2"] = (
        df["ari_attendance"]
        .shift(2)
    )

    df["ari_lag4"] = (
        df["ari_attendance"]
        .shift(4)
    )

    df["ari_recent_2week_mean"] = (
        df["ari_attendance"]
        .shift(1)
        .rolling(
            window=2,
            min_periods=2,
        )
        .mean()
    )

    df["ari_recent_4week_mean"] = (
        df["ari_attendance"]
        .shift(1)
        .rolling(
            window=4,
            min_periods=4,
        )
        .mean()
    )

    # --------------------------------------------------------
    # SEASONAL FEATURES
    # --------------------------------------------------------

    df["season_sin"] = np.sin(
        2 * np.pi
        * df["epi_week"]
        / 52.1775
    )

    df["season_cos"] = np.cos(
        2 * np.pi
        * df["epi_week"]
        / 52.1775
    )

    # --------------------------------------------------------
    # CONSTRUCT FORECAST OBSERVATIONS
    # --------------------------------------------------------

    forecast_frames = []

    for horizon in HORIZONS:

        temp = df.copy()

        temp["horizon_weeks"] = horizon

        temp["target"] = (
            df["ari_attendance"]
            .shift(-horizon)
        )

        temp["target_year"] = (
            df["year"]
            .shift(-horizon)
        )

        temp["target_epi_week"] = (
            df["epi_week"]
            .shift(-horizon)
        )

        forecast_frames.append(temp)

    forecast = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # REMOVE ROWS WITHOUT OBSERVED TARGET
    # --------------------------------------------------------

    forecast = forecast[
        forecast["target"].notna()
    ].copy()

    forecast["target_year"] = (
        forecast["target_year"]
        .astype(int)
    )

    forecast["target_epi_week"] = (
        forecast["target_epi_week"]
        .astype(int)
    )

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    forecast["split"] = (
        forecast["target_year"]
        .apply(assign_split)
    )

    forecast = forecast[
        forecast["split"].notna()
    ].copy()

    # --------------------------------------------------------
    # REQUIRE COMPLETE PREDICTOR HISTORY
    # --------------------------------------------------------

    predictor_columns = [
        "ari_current",
        "ari_lag1",
        "ari_lag2",
        "ari_lag4",
        "ari_recent_2week_mean",
        "ari_recent_4week_mean",
        "season_sin",
        "season_cos",
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

    complete = (
        forecast[predictor_columns]
        .notna()
        .all(axis=1)
    )

    forecast = (
        forecast.loc[complete]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )
    print("FORECAST DATASET QC")
    print("=" * 72)

    print(
        "\nRows by split and horizon:"
    )

    counts = (
        forecast
        .groupby(
            [
                "split",
                "horizon_weeks",
            ]
        )
        .size()
        .rename("n")
        .reset_index()
    )

    print(
        counts.to_string(index=False)
    )

    print(
        "\nRows by target year:"
    )

    print(
        forecast
        .groupby(
            [
                "split",
                "target_year",
                "horizon_weeks",
            ]
        )
        .size()
        .to_string()
    )

    duplicate_count = int(
        forecast.duplicated(
            subset=[
                "year",
                "epi_week",
                "horizon_weeks",
            ]
        ).sum()
    )

    print(
        "\nDuplicate origin/horizon rows:",
        duplicate_count,
    )

    print(
        "Missing targets:",
        int(
            forecast["target"]
            .isna()
            .sum()
        ),
    )

    print(
        "Missing predictor values:",
        int(
            forecast[
                predictor_columns
            ]
            .isna()
            .sum()
            .sum()
        ),
    )

    # --------------------------------------------------------
    # CHECK TARGET ALIGNMENT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )
    print("EXAMPLE 2026 TEST ROWS")
    print("=" * 72)

    sample = (
        forecast[
            forecast["split"] == "test"
        ][
            [
                "year",
                "epi_week",
                "horizon_weeks",
                "target_year",
                "target_epi_week",
                "ari_current",
                "target",
            ]
        ]
        .head(15)
    )

    print(
        sample.to_string(index=False)
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

    forecast.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    counts.to_csv(
        QC_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 72
    )
    print("OUTPUT FILES")
    print("=" * 72)

    print(
        f"\nForecasting dataset:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nQC table:\n"
        f"{QC_FILE}"
    )


if __name__ == "__main__":
    main()