"""
15_validation_model_comparison.py

Compare the best validation models from:

1. Simple baseline forecasts
2. Negative Binomial regression
3. XGBoost
4. A pre-specified 50:50 NB-XGBoost ensemble

Only the 2025 validation period is used.

The 2026 test set is NOT accessed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

ARI_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ari_weekly_clean.csv"
)

BASELINE_METRICS = (
    TABLE_DIR
    / "baseline_forecast_metrics.csv"
)

NB_METRICS = (
    TABLE_DIR
    / "negative_binomial_validation_metrics.csv"
)

NB_PREDICTIONS = (
    TABLE_DIR
    / "negative_binomial_validation_predictions.csv"
)

XGB_METRICS = (
    TABLE_DIR
    / "xgboost_validation_metrics.csv"
)

XGB_PREDICTIONS = (
    TABLE_DIR
    / "xgboost_validation_predictions.csv"
)

COMPARISON_OUTPUT = (
    TABLE_DIR
    / "validation_model_comparison.csv"
)

ENSEMBLE_OUTPUT = (
    TABLE_DIR
    / "validation_ensemble_predictions.csv"
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
    print("VALIDATION MODEL COMPARISON")
    print("=" * 70)

    ari = pd.read_csv(
        ARI_FILE
    )

    baseline_metrics = pd.read_csv(
        BASELINE_METRICS
    )

    nb_metrics = pd.read_csv(
        NB_METRICS
    )

    nb_predictions = pd.read_csv(
        NB_PREDICTIONS
    )

    xgb_metrics = pd.read_csv(
        XGB_METRICS
    )

    xgb_predictions = pd.read_csv(
        XGB_PREDICTIONS
    )

    # ---------------------------------------------------------
    # MASE denominator
    #
    # Keep exactly the same scale used during validation:
    # 2023-2024 training data only.
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

    # ---------------------------------------------------------
    # Best baseline by horizon
    # ---------------------------------------------------------

    baseline_validation = (
        baseline_metrics[
            baseline_metrics["split"]
            == "validation"
        ]
        .copy()
    )

    best_baseline = (
        baseline_validation
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

    # ---------------------------------------------------------
    # Best converged NB by horizon
    # ---------------------------------------------------------

    nb_valid = nb_metrics.copy()

    if "converged" in nb_valid.columns:

        nb_valid = nb_valid[
            nb_valid["converged"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        ].copy()

    best_nb = (
        nb_valid
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

    # ---------------------------------------------------------
    # Best XGBoost by horizon
    # ---------------------------------------------------------

    best_xgb = (
        xgb_metrics
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

    comparison_records = []
    ensemble_records = []

    for horizon in [1, 2, 4]:

        # -----------------------------------------------------
        # Baseline record
        # -----------------------------------------------------

        baseline_row = best_baseline[
            best_baseline["horizon_weeks"]
            == horizon
        ].iloc[0]

        comparison_records.append(
            {
                "horizon_weeks": horizon,
                "model_family": "baseline",
                "specification":
                    baseline_row["model"],
                "n":
                    int(baseline_row["n"]),
                "MAE":
                    float(baseline_row["MAE"]),
                "RMSE":
                    float(baseline_row["RMSE"]),
                "MASE":
                    float(baseline_row["MASE"]),
            }
        )

        # -----------------------------------------------------
        # NB record
        # -----------------------------------------------------

        nb_row = best_nb[
            best_nb["horizon_weeks"]
            == horizon
        ].iloc[0]

        comparison_records.append(
            {
                "horizon_weeks": horizon,
                "model_family":
                    "negative_binomial",
                "specification":
                    nb_row["feature_set"],
                "n":
                    int(
                        nb_row[
                            "n_validation"
                        ]
                    ),
                "MAE":
                    float(nb_row["MAE"]),
                "RMSE":
                    float(nb_row["RMSE"]),
                "MASE":
                    float(nb_row["MASE"]),
            }
        )

        # -----------------------------------------------------
        # XGBoost record
        # -----------------------------------------------------

        xgb_row = best_xgb[
            best_xgb["horizon_weeks"]
            == horizon
        ].iloc[0]

        comparison_records.append(
            {
                "horizon_weeks": horizon,
                "model_family": "xgboost",
                "specification":
                    (
                        str(
                            xgb_row[
                                "feature_set"
                            ]
                        )
                        + "_config_"
                        + str(
                            xgb_row[
                                "config_id"
                            ]
                        )
                    ),
                "n":
                    int(
                        xgb_row[
                            "n_validation"
                        ]
                    ),
                "MAE":
                    float(xgb_row["MAE"]),
                "RMSE":
                    float(xgb_row["RMSE"]),
                "MASE":
                    float(xgb_row["MASE"]),
            }
        )

        # -----------------------------------------------------
        # Pull predictions for selected NB model
        # -----------------------------------------------------

        selected_nb = nb_predictions[
            (
                nb_predictions[
                    "horizon_weeks"
                ]
                == horizon
            )
            &
            (
                nb_predictions[
                    "feature_set"
                ]
                == nb_row[
                    "feature_set"
                ]
            )
        ][
            [
                "target_year",
                "target_epi_week",
                "observed_ari",
                "predicted_ari",
            ]
        ].copy()

        selected_nb = selected_nb.rename(
            columns={
                "predicted_ari":
                    "nb_prediction"
            }
        )

        # -----------------------------------------------------
        # Pull predictions for selected XGBoost model
        # -----------------------------------------------------

        selected_xgb = xgb_predictions[
            (
                xgb_predictions[
                    "horizon_weeks"
                ]
                == horizon
            )
            &
            (
                xgb_predictions[
                    "feature_set"
                ]
                == xgb_row[
                    "feature_set"
                ]
            )
            &
            (
                xgb_predictions[
                    "config_id"
                ].astype(str)
                ==
                str(
                    xgb_row[
                        "config_id"
                    ]
                )
            )
        ][
            [
                "target_year",
                "target_epi_week",
                "observed_ari",
                "predicted_ari",
            ]
        ].copy()

        selected_xgb = selected_xgb.rename(
            columns={
                "predicted_ari":
                    "xgb_prediction"
            }
        )

        # -----------------------------------------------------
        # Merge predictions
        # -----------------------------------------------------

        merged = selected_nb.merge(
            selected_xgb,
            on=[
                "target_year",
                "target_epi_week",
            ],
            how="inner",
            suffixes=(
                "_nb",
                "_xgb",
            ),
        )

        # Ensure observed outcomes agree.
        if not np.allclose(
            merged["observed_ari_nb"],
            merged["observed_ari_xgb"],
        ):

            raise RuntimeError(
                "NB and XGBoost observed "
                "targets do not match."
            )

        merged["observed_ari"] = (
            merged[
                "observed_ari_nb"
            ]
        )

        # -----------------------------------------------------
        # Pre-specified 50:50 ensemble
        # -----------------------------------------------------

        merged["ensemble_prediction"] = (
            0.5
            * merged["nb_prediction"]
            +
            0.5
            * merged["xgb_prediction"]
        )

        merged["horizon_weeks"] = (
            horizon
        )

        ensemble_records.append(
            merged[
                [
                    "horizon_weeks",
                    "target_year",
                    "target_epi_week",
                    "observed_ari",
                    "nb_prediction",
                    "xgb_prediction",
                    "ensemble_prediction",
                ]
            ]
        )

        y_true = (
            merged["observed_ari"]
            .to_numpy(dtype=float)
        )

        y_pred = (
            merged[
                "ensemble_prediction"
            ]
            .to_numpy(dtype=float)
        )

        ensemble_mae = (
            calculate_mae(
                y_true,
                y_pred,
            )
        )

        ensemble_rmse = (
            calculate_rmse(
                y_true,
                y_pred,
            )
        )

        ensemble_mase = (
            ensemble_mae
            / mase_scale
        )

        comparison_records.append(
            {
                "horizon_weeks": horizon,
                "model_family":
                    "nb_xgboost_ensemble",
                "specification":
                    "equal_weight_50_50",
                "n":
                    len(merged),
                "MAE":
                    ensemble_mae,
                "RMSE":
                    ensemble_rmse,
                "MASE":
                    ensemble_mase,
            }
        )

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        comparison_records
    )

    comparison = (
        comparison
        .sort_values(
            [
                "horizon_weeks",
                "MAE",
            ]
        )
        .reset_index(drop=True)
    )

    ensemble_predictions = pd.concat(
        ensemble_records,
        ignore_index=True,
    )

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    ensemble_predictions.to_csv(
        ENSEMBLE_OUTPUT,
        index=False,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print(
        "\nVALIDATION MODEL COMPARISON"
    )

    print(
        comparison
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nBEST VALIDATION MODEL BY HORIZON"
    )

    best = (
        comparison
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
                "model_family",
                "specification",
                "MAE",
                "RMSE",
                "MASE",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        f"\nComparison saved to:\n"
        f"{COMPARISON_OUTPUT}"
    )

    print(
        f"\nEnsemble predictions saved to:\n"
        f"{ENSEMBLE_OUTPUT}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe 2026 test set was NOT accessed."
    )


if __name__ == "__main__":
    main()