"""
17_final_ari_model_comparison.py

Compare frozen final ARI forecasts with simple baselines on the
same 2026 held-out observations using one common MASE denominator.

No model selection is performed here.
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

FINAL_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "final_ari_test_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "final_ari_model_comparison.csv"
)


def mae(y_true, y_pred):
    return np.mean(
        np.abs(y_true - y_pred)
    )


def rmse(y_true, y_pred):
    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def evaluate(
    y_true,
    y_pred,
    mase_scale,
):

    return {
        "MAE": mae(
            y_true,
            y_pred,
        ),
        "RMSE": rmse(
            y_true,
            y_pred,
        ),
        "MASE": (
            mae(y_true, y_pred)
            / mase_scale
        ),
    }


def main():

    print("=" * 70)
    print("FINAL 2026 ARI MODEL COMPARISON")
    print("=" * 70)

    forecast = pd.read_csv(
        FORECAST_FILE
    )

    ari = pd.read_csv(
        ARI_FILE
    )

    final_predictions = pd.read_csv(
        FINAL_PREDICTIONS_FILE
    )

    # --------------------------------------------------------
    # Common MASE scale:
    # all development observations from 2023-2025.
    # --------------------------------------------------------

    development_series = (
        ari[
            ari["year"].isin(
                [2023, 2024, 2025]
            )
        ]
        .sort_values(
            "time_index"
        )[
            "ari_attendance"
        ]
        .astype(float)
    )

    mase_scale = (
        development_series
        .diff()
        .abs()
        .dropna()
        .mean()
    )

    print(
        f"\nCommon MASE denominator: "
        f"{mase_scale:.3f}"
    )

    records = []

    for horizon in [
        1,
        2,
        4,
    ]:

        test = forecast[
            (
                forecast["split"]
                == "test"
            )
            &
            (
                forecast[
                    "horizon_weeks"
                ]
                == horizon
            )
        ].copy()

        final = final_predictions[
            final_predictions[
                "horizon_weeks"
            ]
            == horizon
        ].copy()

        merged = test.merge(
            final[
                [
                    "target_year",
                    "target_epi_week",
                    "nb_prediction",
                    "xgb_prediction",
                    "ensemble_prediction",
                    "final_prediction",
                    "selected_model",
                ]
            ],
            on=[
                "target_year",
                "target_epi_week",
            ],
            how="inner",
        )

        # ----------------------------------------------------
        # Recreate simple baselines.
        # ----------------------------------------------------

        merged[
            "persistence"
        ] = merged[
            "ari_current"
        ]

        merged[
            "recent_4week_mean"
        ] = (
            merged[
                [
                    "ari_current",
                    "ari_lag_1",
                    "ari_lag_2",
                    "ari_lag_3",
                ]
            ]
            .mean(axis=1)
        )

        merged[
            "seasonal_naive"
        ] = merged[
            "ari_target_same_week_previous_year"
        ]

        candidates = {

            "persistence":
                "persistence",

            "recent_4week_mean":
                "recent_4week_mean",

            "seasonal_naive":
                "seasonal_naive",

            "negative_binomial":
                "nb_prediction",

            "xgboost":
                "xgb_prediction",

            "nb_xgboost_ensemble":
                "ensemble_prediction",

            "FINAL_SELECTED_MODEL":
                "final_prediction",
        }

        for (
            model_name,
            prediction_column,
        ) in candidates.items():

            valid = merged[
                [
                    "target_ari",
                    prediction_column,
                ]
            ].dropna()

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

            metrics = evaluate(
                y_true,
                y_pred,
                mase_scale,
            )

            records.append(
                {
                    "horizon_weeks":
                        horizon,

                    "model":
                        model_name,

                    "n":
                        len(valid),

                    "MAE":
                        metrics["MAE"],

                    "RMSE":
                        metrics["RMSE"],

                    "MASE":
                        metrics["MASE"],
                }
            )

    results = pd.DataFrame(
        records
    )

    results = (
        results
        .sort_values(
            [
                "horizon_weeks",
                "MAE",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nFINAL COMPARISON"
    )

    print(
        results
        .round(3)
        .to_string(
            index=False
        )
    )

    print(
        "\nBEST OBSERVED 2026 PERFORMANCE "
        "(DESCRIPTIVE ONLY)"
    )

    best = (
        results[
            results["model"]
            != "FINAL_SELECTED_MODEL"
        ]
        .sort_values(
            [
                "horizon_weeks",
                "MAE",
            ]
        )
        .groupby(
            "horizon_weeks",
            as_index=False,
        )
        .first()
    )

    print(
        best[
            [
                "horizon_weeks",
                "model",
                "MAE",
                "RMSE",
                "MASE",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    print(
        "\nIMPORTANT:"
        "\nThese descriptive comparisons must NOT "
        "be used to change the frozen model."
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()