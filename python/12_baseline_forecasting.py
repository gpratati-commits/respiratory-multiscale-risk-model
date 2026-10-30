"""
12_baseline_forecasting.py

Evaluate simple ARI forecasting baselines for
1-, 2-, and 4-week-ahead prediction.

Baselines
---------
1. Persistence
   Forecast = current ARI attendance.

2. Recent 4-week mean
   Forecast = mean of current ARI and previous 3 weeks.

3. Seasonal naive
   Forecast = ARI attendance from the same TARGET
   epidemiological week in the previous year.

Metrics
-------
MAE
RMSE
MASE

Important:
The MASE scale is calculated using the training period only.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ari_forecasting_dataset.csv"
)

ARI_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ari_weekly_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "baseline_forecast_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "baseline_forecast_predictions.csv"
)


def calculate_mae(y_true, y_pred):
    return np.mean(
        np.abs(
            y_true - y_pred
        )
    )


def calculate_rmse(y_true, y_pred):
    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def main():

    print("=" * 70)
    print("ARI BASELINE FORECASTING")
    print("=" * 70)

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n"
            f"{FORECAST_FILE}"
        )

    if not ARI_FILE.exists():
        raise FileNotFoundError(
            f"ARI dataset not found:\n"
            f"{ARI_FILE}"
        )

    forecast = pd.read_csv(
        FORECAST_FILE
    )

    ari = pd.read_csv(
        ARI_FILE
    )

    # ---------------------------------------------------------
    # MASE scaling denominator
    #
    # Use training data only: 2023-2024.
    # ---------------------------------------------------------

    train_series = (
        ari[
            ari["year"].isin(
                [2023, 2024]
            )
        ]
        .sort_values(
            "time_index"
        )["ari_attendance"]
        .astype(float)
    )

    mase_scale = (
        train_series
        .diff()
        .abs()
        .dropna()
        .mean()
    )

    if (
        pd.isna(mase_scale)
        or mase_scale <= 0
    ):
        raise RuntimeError(
            "Invalid MASE scaling denominator."
        )

    print(
        f"\nTraining MASE scale: "
        f"{mase_scale:.3f}"
    )

    # ---------------------------------------------------------
    # Baseline 1: persistence
    # ---------------------------------------------------------

    forecast[
        "pred_persistence"
    ] = forecast[
        "ari_current"
    ]

    # ---------------------------------------------------------
    # Baseline 2: recent 4-week mean
    #
    # Uses only information available at the forecast origin.
    # ---------------------------------------------------------

    forecast[
        "pred_recent_4week_mean"
    ] = (
        forecast[
            [
                "ari_current",
                "ari_lag_1",
                "ari_lag_2",
                "ari_lag_3",
            ]
        ]
        .mean(axis=1)
    )

    # ---------------------------------------------------------
    # Baseline 3: seasonal naive
    #
    # Uses same TARGET epidemiological week
    # from previous year.
    # ---------------------------------------------------------

    forecast[
        "pred_seasonal_naive"
    ] = forecast[
        "ari_target_same_week_previous_year"
    ]

    prediction_columns = {
        "persistence":
            "pred_persistence",

        "recent_4week_mean":
            "pred_recent_4week_mean",

        "seasonal_naive":
            "pred_seasonal_naive",
    }

    metric_records = []

    # ---------------------------------------------------------
    # Evaluate each model separately by split and horizon.
    # ---------------------------------------------------------

    for split in [
        "train",
        "validation",
        "test",
    ]:

        for horizon in [
            1,
            2,
            4,
        ]:

            subset = forecast[
                (
                    forecast["split"]
                    == split
                )
                &
                (
                    forecast[
                        "horizon_weeks"
                    ]
                    == horizon
                )
            ].copy()

            for (
                model_name,
                prediction_column,
            ) in prediction_columns.items():

                valid = subset[
                    [
                        "target_ari",
                        prediction_column,
                    ]
                ].dropna()

                if valid.empty:
                    continue

                y_true = (
                    valid[
                        "target_ari"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                y_pred = (
                    valid[
                        prediction_column
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                mae = calculate_mae(
                    y_true,
                    y_pred,
                )

                rmse = calculate_rmse(
                    y_true,
                    y_pred,
                )

                mase = (
                    mae
                    / mase_scale
                )

                metric_records.append(
                    {
                        "split":
                            split,

                        "horizon_weeks":
                            horizon,

                        "model":
                            model_name,

                        "n":
                            len(valid),

                        "MAE":
                            mae,

                        "RMSE":
                            rmse,

                        "MASE":
                            mase,
                    }
                )

    metrics = pd.DataFrame(
        metric_records
    )

    metrics = (
        metrics
        .sort_values(
            [
                "split",
                "horizon_weeks",
                "MAE",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    forecast.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Print validation results.
    #
    # These are the results we use for model-development
    # decisions.
    # ---------------------------------------------------------

    validation = metrics[
        metrics["split"]
        == "validation"
    ]

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS — 2025")
    print("=" * 70)

    print(
        validation
        .round(3)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Print test results for record keeping.
    #
    # IMPORTANT:
    # Do not use these results for tuning future models.
    # ---------------------------------------------------------

    test = metrics[
        metrics["split"]
        == "test"
    ]

    print("\n" + "=" * 70)
    print("HELD-OUT TEST RESULTS — 2026")
    print("=" * 70)

    print(
        test
        .round(3)
        .to_string(index=False)
    )

    print(
        f"\nMetrics saved to:\n"
        f"{METRICS_FILE}"
    )

    print(
        f"\nPredictions saved to:\n"
        f"{PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()