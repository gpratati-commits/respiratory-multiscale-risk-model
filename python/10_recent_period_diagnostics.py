"""
10_recent_period_diagnostics.py

Repeat key ARI diagnostics using the recent forecasting period.

Primary recent period:
    2023 through 2026

Seasonality estimation:
    2023 through 2025 only,
    because 2026 is currently incomplete.

Outputs:
    outputs/tables/ari_recent_autocorrelation.csv
    outputs/tables/ari_recent_seasonal_profile.csv

    outputs/figures/ari_recent_diagnostics/
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ari_weekly_clean.csv"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "ari_recent_diagnostics"
)

ACF_OUTPUT = (
    TABLE_DIR
    / "ari_recent_autocorrelation.csv"
)

SEASONAL_OUTPUT = (
    TABLE_DIR
    / "ari_recent_seasonal_profile.csv"
)


def main():

    print("=" * 70)
    print("RECENT-PERIOD ARI DIAGNOSTICS")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df = (
        df.sort_values(
            ["year", "epi_week"]
        )
        .reset_index(drop=True)
    )

    recent = df[
        df["year"] >= 2023
    ].copy()

    print(
        "\nRecent-period observations:",
        len(recent),
    )

    print("\nObservations by year:")

    print(
        recent.groupby("year").size()
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Recent-period autocorrelation
    # ---------------------------------------------------------

    series = (
        recent["ari_attendance"]
        .astype(float)
        .reset_index(drop=True)
    )

    records = []

    for lag in range(1, 53):

        records.append(
            {
                "lag_weeks": lag,
                "autocorrelation":
                    series.autocorr(lag=lag),
            }
        )

    acf = pd.DataFrame(records)

    acf.to_csv(
        ACF_OUTPUT,
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
        "Autocorrelation of ARI Attendance, 2023-2026"
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
        / "ari_recent_autocorrelation.png",
        dpi=300,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Recent seasonal pattern
    #
    # 2026 excluded because the year is incomplete.
    # ---------------------------------------------------------

    seasonal_source = df[
        df["year"].between(
            2023,
            2025,
        )
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
        SEASONAL_OUTPUT,
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
        "Recent Seasonal Pattern of ARI Attendance, 2023-2025"
    )

    ax.set_xlabel(
        "Epidemiological week"
    )

    ax.set_ylabel(
        "Average daily ARI attendance"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "ari_recent_seasonal_profile.png",
        dpi=300,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Print key lags
    # ---------------------------------------------------------

    key_lags = [
        1,
        2,
        3,
        4,
        8,
        13,
        26,
        52,
    ]

    print("\nKEY AUTOCORRELATION LAGS")

    print(
        acf[
            acf["lag_weeks"]
            .isin(key_lags)
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nSTRONGEST 10 LAGS")

    strongest = (
        acf.assign(
            abs_correlation=
            acf["autocorrelation"].abs()
        )
        .nlargest(
            10,
            "abs_correlation",
        )
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

    print(ACF_OUTPUT)
    print(SEASONAL_OUTPUT)
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()