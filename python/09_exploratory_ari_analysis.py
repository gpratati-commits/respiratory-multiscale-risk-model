"""
09_exploratory_ari_analysis.py

Exploratory analysis of Singapore weekly ARI polyclinic attendance.

Input
-----
data/processed/ari_weekly_clean.csv

Outputs
-------
outputs/figures/ari_eda/
outputs/tables/ari_yearly_summary.csv
outputs/tables/ari_seasonal_profile.csv
outputs/tables/ari_autocorrelation.csv

This stage is descriptive only.
No forecast model is fitted here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ari_weekly_clean.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "ari_eda"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

YEAR_SUMMARY_FILE = (
    TABLE_DIR
    / "ari_yearly_summary.csv"
)

SEASONAL_FILE = (
    TABLE_DIR
    / "ari_seasonal_profile.csv"
)

ACF_FILE = (
    TABLE_DIR
    / "ari_autocorrelation.csv"
)


def main():

    print("=" * 70)
    print("EXPLORATORY ANALYSIS OF WEEKLY ARI ATTENDANCE")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "year",
        "epi_week",
        "time_index",
        "ari_attendance",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = (
        df.sort_values("time_index")
        .reset_index(drop=True)
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Yearly descriptive statistics
    # ---------------------------------------------------------

    yearly = (
        df.groupby("year")["ari_attendance"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    yearly.to_csv(
        YEAR_SUMMARY_FILE,
        index=False,
    )

    print("\nYEARLY SUMMARY")
    print(yearly.round(1).to_string(index=False))

    # ---------------------------------------------------------
    # 2. Full time-series figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    ax.plot(
        df["time_index"],
        df["ari_attendance"],
        linewidth=1.2,
    )

    year_starts = (
        df.groupby("year")
        .first()
        .reset_index()
    )

    for _, row in year_starts.iterrows():

        ax.axvline(
            row["time_index"],
            linewidth=0.6,
            alpha=0.4,
        )

    ax.set_title(
        "Weekly ARI Polyclinic Attendance in Singapore"
    )

    ax.set_xlabel(
        "Sequential epidemiological week"
    )

    ax.set_ylabel(
        "Average daily ARI polyclinic attendance"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "ari_full_time_series.png",
        dpi=300,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # 3. Yearly distributions
    # ---------------------------------------------------------

    years = sorted(
        df["year"].unique()
    )

    values_by_year = [
        df.loc[
            df["year"] == year,
            "ari_attendance",
        ].values
        for year in years
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.boxplot(
        values_by_year,
        tick_labels=[
            str(year)
            for year in years
        ],
    )

    ax.set_title(
        "Distribution of Weekly ARI Attendance by Year"
    )

    ax.set_xlabel("Year")

    ax.set_ylabel(
        "Average daily ARI polyclinic attendance"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "ari_yearly_distribution.png",
        dpi=300,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # 4. Seasonal profile
    #
    # 2026 is excluded because it is currently incomplete.
    # ---------------------------------------------------------

    seasonal_source = df[
        df["year"] <= 2025
    ].copy()

    seasonal = (
        seasonal_source
        .groupby("epi_week")[
            "ari_attendance"
        ]
        .agg(
            observations="count",
            mean="mean",
            median="median",
            std="std",
        )
        .reset_index()
    )

    seasonal.to_csv(
        SEASONAL_FILE,
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        seasonal["epi_week"],
        seasonal["mean"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Mean",
    )

    ax.plot(
        seasonal["epi_week"],
        seasonal["median"],
        linewidth=1,
        label="Median",
    )

    ax.set_title(
        "Average Seasonal Pattern of ARI Attendance"
    )

    ax.set_xlabel(
        "Epidemiological week"
    )

    ax.set_ylabel(
        "ARI attendance"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "ari_seasonal_profile.png",
        dpi=300,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # 5. Autocorrelation
    # ---------------------------------------------------------

    series = (
        df["ari_attendance"]
        .astype(float)
    )

    acf_records = []

    max_lag = min(
        52,
        len(series) - 2,
    )

    for lag in range(
        1,
        max_lag + 1,
    ):

        correlation = (
            series.autocorr(
                lag=lag
            )
        )

        acf_records.append(
            {
                "lag_weeks": lag,
                "autocorrelation": correlation,
            }
        )

    acf = pd.DataFrame(
        acf_records
    )

    acf.to_csv(
        ACF_FILE,
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.stem(
        acf["lag_weeks"],
        acf["autocorrelation"],
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_title(
        "Autocorrelation of Weekly ARI Attendance"
    )

    ax.set_xlabel(
        "Lag (weeks)"
    )

    ax.set_ylabel(
        "Autocorrelation"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "ari_autocorrelation.png",
        dpi=300,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # 6. Largest sequential changes
    # ---------------------------------------------------------

    df["weekly_change"] = (
        df["ari_attendance"]
        .diff()
    )

    df["absolute_change"] = (
        df["weekly_change"]
        .abs()
    )

    largest_changes = (
        df.nlargest(
            10,
            "absolute_change",
        )[
            [
                "year",
                "epi_week",
                "ari_attendance",
                "weekly_change",
                "absolute_change",
            ]
        ]
    )

    print(
        "\nLARGEST WEEK-TO-WEEK CHANGES"
    )

    print(
        largest_changes
        .round(1)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 7. Most informative autocorrelation lags
    # ---------------------------------------------------------

    strongest = (
        acf.assign(
            absolute_correlation=
            acf[
                "autocorrelation"
            ].abs()
        )
        .nlargest(
            10,
            "absolute_correlation",
        )
    )

    print(
        "\nSTRONGEST AUTOCORRELATION LAGS"
    )

    print(
        strongest[
            [
                "lag_weeks",
                "autocorrelation",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nOutputs created:")

    print(
        f"- {YEAR_SUMMARY_FILE}"
    )

    print(
        f"- {SEASONAL_FILE}"
    )

    print(
        f"- {ACF_FILE}"
    )

    print(
        f"- {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()