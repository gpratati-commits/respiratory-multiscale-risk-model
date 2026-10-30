"""
16_final_ari_test_evaluation.py

Final held-out evaluation of the frozen ARI forecasting models.

Development data:
    2023-2025

Held-out test:
    2026

Frozen model strategy:
    1-week -> 50:50 Negative Binomial + XGBoost ensemble
    2-week -> XGBoost
    4-week -> 50:50 Negative Binomial + XGBoost ensemble

IMPORTANT
---------
No model selection or hyperparameter tuning is performed here.

All choices were frozen using the 2025 validation period before
the 2026 test set was evaluated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import NegativeBinomial
from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

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

XGB_CONFIG_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "xgboost_best_configurations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "final_ari_test_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "final_ari_test_predictions.csv"
)


# ============================================================
# LOCKED FEATURES
# ============================================================

NB_FEATURES = [
    "ari_current",
    "ari_lag_1",
    "ari_lag_2",
    "season_sin_1",
    "season_cos_1",
]

XGB_FEATURES = [
    "ari_current",
    "ari_lag_1",
    "ari_lag_2",
    "ari_lag_3",
    "ari_lag_4",
    "season_sin_1",
    "season_cos_1",
]


# ============================================================
# LOCKED FINAL MODEL BY HORIZON
# ============================================================

FINAL_MODEL = {
    1: "ensemble",
    2: "xgboost",
    4: "ensemble",
}


# ============================================================
# METRICS
# ============================================================

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


def evaluate(
    y_true,
    y_pred,
    mase_scale,
):

    model_mae = calculate_mae(
        y_true,
        y_pred,
    )

    model_rmse = calculate_rmse(
        y_true,
        y_pred,
    )

    model_mase = (
        model_mae
        / mase_scale
    )

    return (
        model_mae,
        model_rmse,
        model_mase,
    )


# ============================================================
# STANDARDIZATION FOR NEGATIVE BINOMIAL
# ============================================================

def standardize(
    development,
    test,
    features,
):

    development_x = (
        development[
            features
        ]
        .astype(float)
        .copy()
    )

    test_x = (
        test[
            features
        ]
        .astype(float)
        .copy()
    )

    means = (
        development_x
        .mean()
    )

    stds = (
        development_x
        .std()
        .replace(
            0,
            1.0,
        )
    )

    development_scaled = (
        development_x
        - means
    ) / stds

    test_scaled = (
        test_x
        - means
    ) / stds

    return (
        development_scaled,
        test_scaled,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL HELD-OUT ARI TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    forecast = pd.read_csv(
        FORECAST_FILE
    )

    ari = pd.read_csv(
        ARI_FILE
    )

    xgb_configs = pd.read_csv(
        XGB_CONFIG_FILE
    )

    # --------------------------------------------------------
    # MASE denominator
    #
    # Final models are refitted using all development data:
    # 2023-2025.
    #
    # Therefore the final test MASE denominator is also
    # calculated from development data only.
    # --------------------------------------------------------

    development_ari = (
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
        development_ari
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
            "Invalid final MASE denominator."
        )

    print(
        f"\nDevelopment MASE scale: "
        f"{mase_scale:.3f}"
    )

    metric_records = []
    prediction_records = []

    # --------------------------------------------------------
    # Evaluate each horizon
    # --------------------------------------------------------

    for horizon in [
        1,
        2,
        4,
    ]:

        print("\n" + "=" * 70)
        print(
            f"HORIZON: {horizon} WEEK(S)"
        )
        print("=" * 70)

        horizon_data = forecast[
            forecast[
                "horizon_weeks"
            ]
            == horizon
        ].copy()

        # Development = original training + validation.
        development = horizon_data[
            horizon_data["split"]
            .isin(
                [
                    "train",
                    "validation",
                ]
            )
        ].copy()

        # Test = untouched 2026 targets.
        test = horizon_data[
            horizon_data["split"]
            == "test"
        ].copy()

        if development.empty:
            raise RuntimeError(
                f"No development data "
                f"for horizon {horizon}."
            )

        if test.empty:
            raise RuntimeError(
                f"No test data "
                f"for horizon {horizon}."
            )

        # ----------------------------------------------------
        # NEGATIVE BINOMIAL
        # ----------------------------------------------------

        nb_development = (
            development
            .dropna(
                subset=
                NB_FEATURES
                + ["target_ari"]
            )
            .copy()
        )

        nb_test = (
            test
            .dropna(
                subset=
                NB_FEATURES
                + ["target_ari"]
            )
            .copy()
        )

        (
            nb_x_development,
            nb_x_test,
        ) = standardize(
            development=
                nb_development,
            test=
                nb_test,
            features=
                NB_FEATURES,
        )

        nb_x_development = (
            sm.add_constant(
                nb_x_development,
                has_constant="add",
            )
        )

        nb_x_test = (
            sm.add_constant(
                nb_x_test,
                has_constant="add",
            )
        )

        nb_y_development = (
            nb_development[
                "target_ari"
            ]
            .astype(float)
            .to_numpy()
        )

        nb_model = NegativeBinomial(
            nb_y_development,
            nb_x_development,
            loglike_method="nb2",
        )

        nb_result = nb_model.fit(
            disp=False,
            maxiter=500,
        )

        nb_prediction = np.asarray(
            nb_result.predict(
                nb_x_test
            )
        )

        nb_prediction = np.maximum(
            nb_prediction,
            0,
        )

        # ----------------------------------------------------
        # XGBOOST
        #
        # Hyperparameters were selected using 2025 validation.
        # They are read from the saved configuration file.
        # ----------------------------------------------------

        config_row = (
            xgb_configs[
                xgb_configs[
                    "horizon_weeks"
                ]
                == horizon
            ]
            .iloc[0]
        )

        xgb_development = (
            development
            .dropna(
                subset=
                XGB_FEATURES
                + ["target_ari"]
            )
            .copy()
        )

        xgb_test = (
            test
            .dropna(
                subset=
                XGB_FEATURES
                + ["target_ari"]
            )
            .copy()
        )

        xgb_model = XGBRegressor(

            objective=
                "reg:squarederror",

            eval_metric=
                "rmse",

            n_estimators=
                int(
                    config_row[
                        "n_estimators"
                    ]
                ),

            max_depth=
                int(
                    config_row[
                        "max_depth"
                    ]
                ),

            learning_rate=
                float(
                    config_row[
                        "learning_rate"
                    ]
                ),

            min_child_weight=
                float(
                    config_row[
                        "min_child_weight"
                    ]
                ),

            subsample=
                float(
                    config_row[
                        "subsample"
                    ]
                ),

            colsample_bytree=
                float(
                    config_row[
                        "colsample_bytree"
                    ]
                ),

            reg_lambda=
                float(
                    config_row[
                        "reg_lambda"
                    ]
                ),

            random_state=42,

            n_jobs=2,
        )

        xgb_model.fit(
            xgb_development[
                XGB_FEATURES
            ].astype(float),

            xgb_development[
                "target_ari"
            ].astype(float),
        )

        xgb_prediction = (
            xgb_model.predict(
                xgb_test[
                    XGB_FEATURES
                ].astype(float)
            )
        )

        xgb_prediction = np.maximum(
            xgb_prediction,
            0,
        )

        # ----------------------------------------------------
        # Align test rows
        # ----------------------------------------------------

        nb_pred_df = pd.DataFrame(
            {
                "target_year":
                    nb_test[
                        "target_year"
                    ].astype(int)
                    .to_numpy(),

                "target_epi_week":
                    nb_test[
                        "target_epi_week"
                    ].astype(int)
                    .to_numpy(),

                "target_ari":
                    nb_test[
                        "target_ari"
                    ].astype(float)
                    .to_numpy(),

                "nb_prediction":
                    nb_prediction,
            }
        )

        xgb_pred_df = pd.DataFrame(
            {
                "target_year":
                    xgb_test[
                        "target_year"
                    ].astype(int)
                    .to_numpy(),

                "target_epi_week":
                    xgb_test[
                        "target_epi_week"
                    ].astype(int)
                    .to_numpy(),

                "target_ari":
                    xgb_test[
                        "target_ari"
                    ].astype(float)
                    .to_numpy(),

                "xgb_prediction":
                    xgb_prediction,
            }
        )

        merged = nb_pred_df.merge(
            xgb_pred_df,
            on=[
                "target_year",
                "target_epi_week",
            ],
            suffixes=(
                "_nb",
                "_xgb",
            ),
            how="inner",
        )

        if not np.allclose(
            merged[
                "target_ari_nb"
            ],
            merged[
                "target_ari_xgb"
            ],
        ):

            raise RuntimeError(
                "NB and XGBoost test "
                "targets do not match."
            )

        merged[
            "observed_ari"
        ] = merged[
            "target_ari_nb"
        ]

        # ----------------------------------------------------
        # Frozen equal-weight ensemble
        # ----------------------------------------------------

        merged[
            "ensemble_prediction"
        ] = (
            0.5
            * merged[
                "nb_prediction"
            ]
            +
            0.5
            * merged[
                "xgb_prediction"
            ]
        )

        # ----------------------------------------------------
        # Frozen final strategy
        # ----------------------------------------------------

        selected_model = (
            FINAL_MODEL[
                horizon
            ]
        )

        if (
            selected_model
            == "ensemble"
        ):

            merged[
                "final_prediction"
            ] = merged[
                "ensemble_prediction"
            ]

        elif (
            selected_model
            == "xgboost"
        ):

            merged[
                "final_prediction"
            ] = merged[
                "xgb_prediction"
            ]

        else:

            raise RuntimeError(
                "Unknown frozen final model."
            )

        merged[
            "horizon_weeks"
        ] = horizon

        merged[
            "selected_model"
        ] = selected_model

        # ----------------------------------------------------
        # Evaluate NB
        # ----------------------------------------------------

        y_true = (
            merged[
                "observed_ari"
            ]
            .to_numpy(dtype=float)
        )

        nb_pred = (
            merged[
                "nb_prediction"
            ]
            .to_numpy(dtype=float)
        )

        (
            nb_mae,
            nb_rmse,
            nb_mase,
        ) = evaluate(
            y_true,
            nb_pred,
            mase_scale,
        )

        metric_records.append(
            {
                "horizon_weeks":
                    horizon,

                "model":
                    "negative_binomial",

                "n":
                    len(merged),

                "MAE":
                    nb_mae,

                "RMSE":
                    nb_rmse,

                "MASE":
                    nb_mase,
            }
        )

        # ----------------------------------------------------
        # Evaluate XGBoost
        # ----------------------------------------------------

        xgb_pred = (
            merged[
                "xgb_prediction"
            ]
            .to_numpy(dtype=float)
        )

        (
            xgb_mae,
            xgb_rmse,
            xgb_mase,
        ) = evaluate(
            y_true,
            xgb_pred,
            mase_scale,
        )

        metric_records.append(
            {
                "horizon_weeks":
                    horizon,

                "model":
                    "xgboost",

                "n":
                    len(merged),

                "MAE":
                    xgb_mae,

                "RMSE":
                    xgb_rmse,

                "MASE":
                    xgb_mase,
            }
        )

        # ----------------------------------------------------
        # Evaluate ensemble
        # ----------------------------------------------------

        ensemble_pred = (
            merged[
                "ensemble_prediction"
            ]
            .to_numpy(dtype=float)
        )

        (
            ensemble_mae,
            ensemble_rmse,
            ensemble_mase,
        ) = evaluate(
            y_true,
            ensemble_pred,
            mase_scale,
        )

        metric_records.append(
            {
                "horizon_weeks":
                    horizon,

                "model":
                    "nb_xgboost_ensemble",

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

        # ----------------------------------------------------
        # Evaluate frozen final model
        # ----------------------------------------------------

        final_pred = (
            merged[
                "final_prediction"
            ]
            .to_numpy(dtype=float)
        )

        (
            final_mae,
            final_rmse,
            final_mase,
        ) = evaluate(
            y_true,
            final_pred,
            mase_scale,
        )

        metric_records.append(
            {
                "horizon_weeks":
                    horizon,

                "model":
                    "FINAL_SELECTED_MODEL",

                "n":
                    len(merged),

                "MAE":
                    final_mae,

                "RMSE":
                    final_rmse,

                "MASE":
                    final_mase,
            }
        )

        prediction_records.append(
            merged[
                [
                    "horizon_weeks",
                    "target_year",
                    "target_epi_week",
                    "observed_ari",
                    "nb_prediction",
                    "xgb_prediction",
                    "ensemble_prediction",
                    "selected_model",
                    "final_prediction",
                ]
            ]
        )

        print(
            f"\nFrozen final model: "
            f"{selected_model}"
        )

        print(
            f"Final test MAE: "
            f"{final_mae:.3f}"
        )

    # --------------------------------------------------------
    # Combine outputs
    # --------------------------------------------------------

    metrics = pd.DataFrame(
        metric_records
    )

    predictions = pd.concat(
        prediction_records,
        ignore_index=True,
    )

    metrics = (
        metrics
        .sort_values(
            [
                "horizon_weeks",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print final held-out results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST RESULTS — 2026")
    print("=" * 70)

    print(
        metrics
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nFROZEN FINAL MODEL RESULTS"
    )

    final_metrics = metrics[
        metrics["model"]
        == "FINAL_SELECTED_MODEL"
    ]

    print(
        final_metrics[
            [
                "horizon_weeks",
                "n",
                "MAE",
                "RMSE",
                "MASE",
            ]
        ]
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