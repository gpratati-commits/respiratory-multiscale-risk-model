"""
14_xgboost_forecasting.py

Validation-only model selection for XGBoost ARI forecasting.

Training:
    2023-2024

Validation:
    2025

Held-out test:
    2026

IMPORTANT:
The 2026 test period is deliberately NOT evaluated here.

Forecast horizons:
    1 week
    2 weeks
    4 weeks
"""

from pathlib import Path

import numpy as np
import pandas as pd
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "xgboost_validation_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "xgboost_validation_predictions.csv"
)

BEST_CONFIG_FILE = (
    OUTPUT_DIR
    / "xgboost_best_configurations.csv"
)


# ============================================================
# CANDIDATE FEATURE SETS
# ============================================================

FEATURE_SETS = {

    "current_only": [
        "ari_current",
    ],

    "recent_history": [
        "ari_current",
        "ari_lag_1",
        "ari_lag_2",
    ],

    "recent_plus_seasonality": [
        "ari_current",
        "ari_lag_1",
        "ari_lag_2",
        "season_sin_1",
        "season_cos_1",
    ],

    "recent_plus_two_harmonics": [
        "ari_current",
        "ari_lag_1",
        "ari_lag_2",
        "season_sin_1",
        "season_cos_1",
        "season_sin_2",
        "season_cos_2",
    ],

    "full_recent_history_plus_seasonality": [
        "ari_current",
        "ari_lag_1",
        "ari_lag_2",
        "ari_lag_3",
        "ari_lag_4",
        "season_sin_1",
        "season_cos_1",
    ],
}


# ============================================================
# SMALL, CONSERVATIVE HYPERPARAMETER GRID
#
# Dataset is small, so deliberately avoid a large grid.
# ============================================================

PARAMETER_CONFIGS = [

    {
        "config_id": "A",
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_child_weight": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },

    {
        "config_id": "B",
        "n_estimators": 150,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
    },

    {
        "config_id": "C",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
    },

    {
        "config_id": "D",
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    },

    {
        "config_id": "E",
        "n_estimators": 80,
        "max_depth": 1,
        "learning_rate": 0.08,
        "min_child_weight": 4,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 2.0,
    },

    {
        "config_id": "F",
        "n_estimators": 150,
        "max_depth": 1,
        "learning_rate": 0.05,
        "min_child_weight": 6,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "reg_lambda": 5.0,
    },
]


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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XGBOOST ARI FORECASTING")
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

    # --------------------------------------------------------
    # MASE denominator from training data only.
    # --------------------------------------------------------

    train_ari = (
        ari[
            ari["year"].isin(
                [2023, 2024]
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
        train_ari
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
            "Invalid MASE denominator."
        )

    print(
        f"\nTraining MASE scale: "
        f"{mase_scale:.3f}"
    )

    metric_records = []
    prediction_records = []

    # --------------------------------------------------------
    # Evaluate each horizon separately.
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
            forecast["horizon_weeks"]
            == horizon
        ].copy()

        for (
            feature_set_name,
            features,
        ) in FEATURE_SETS.items():

            usable = horizon_data.dropna(
                subset=
                features
                + ["target_ari"]
            ).copy()

            train = usable[
                usable["split"]
                == "train"
            ].copy()

            validation = usable[
                usable["split"]
                == "validation"
            ].copy()

            if (
                train.empty
                or validation.empty
            ):
                continue

            x_train = (
                train[
                    features
                ]
                .astype(float)
            )

            y_train = (
                train[
                    "target_ari"
                ]
                .astype(float)
            )

            x_validation = (
                validation[
                    features
                ]
                .astype(float)
            )

            y_validation = (
                validation[
                    "target_ari"
                ]
                .astype(float)
                .to_numpy()
            )

            # ------------------------------------------------
            # Test each conservative parameter configuration.
            # ------------------------------------------------

            for config in PARAMETER_CONFIGS:

                model = XGBRegressor(

                    objective=
                        "reg:squarederror",

                    eval_metric=
                        "rmse",

                    n_estimators=
                        config[
                            "n_estimators"
                        ],

                    max_depth=
                        config[
                            "max_depth"
                        ],

                    learning_rate=
                        config[
                            "learning_rate"
                        ],

                    min_child_weight=
                        config[
                            "min_child_weight"
                        ],

                    subsample=
                        config[
                            "subsample"
                        ],

                    colsample_bytree=
                        config[
                            "colsample_bytree"
                        ],

                    reg_lambda=
                        config[
                            "reg_lambda"
                        ],

                    random_state=42,

                    n_jobs=2,
                )

                model.fit(
                    x_train,
                    y_train,
                )

                prediction = (
                    model.predict(
                        x_validation
                    )
                )

                # ARI cannot be negative.
                prediction = np.maximum(
                    prediction,
                    0,
                )

                model_mae = (
                    calculate_mae(
                        y_validation,
                        prediction,
                    )
                )

                model_rmse = (
                    calculate_rmse(
                        y_validation,
                        prediction,
                    )
                )

                model_mase = (
                    model_mae
                    / mase_scale
                )

                metric_records.append(
                    {
                        "horizon_weeks":
                            horizon,

                        "feature_set":
                            feature_set_name,

                        "config_id":
                            config[
                                "config_id"
                            ],

                        "n_train":
                            len(train),

                        "n_validation":
                            len(validation),

                        "MAE":
                            model_mae,

                        "RMSE":
                            model_rmse,

                        "MASE":
                            model_mase,

                        "n_estimators":
                            config[
                                "n_estimators"
                            ],

                        "max_depth":
                            config[
                                "max_depth"
                            ],

                        "learning_rate":
                            config[
                                "learning_rate"
                            ],

                        "min_child_weight":
                            config[
                                "min_child_weight"
                            ],

                        "subsample":
                            config[
                                "subsample"
                            ],

                        "colsample_bytree":
                            config[
                                "colsample_bytree"
                            ],

                        "reg_lambda":
                            config[
                                "reg_lambda"
                            ],
                    }
                )

                for j in range(
                    len(validation)
                ):

                    row = (
                        validation
                        .iloc[j]
                    )

                    prediction_records.append(
                        {
                            "horizon_weeks":
                                horizon,

                            "feature_set":
                                feature_set_name,

                            "config_id":
                                config[
                                    "config_id"
                                ],

                            "target_year":
                                int(
                                    row[
                                        "target_year"
                                    ]
                                ),

                            "target_epi_week":
                                int(
                                    row[
                                        "target_epi_week"
                                    ]
                                ),

                            "observed_ari":
                                float(
                                    y_validation[j]
                                ),

                            "predicted_ari":
                                float(
                                    prediction[j]
                                ),
                        }
                    )

    # --------------------------------------------------------
    # Save all validation results.
    # --------------------------------------------------------

    metrics = pd.DataFrame(
        metric_records
    )

    predictions = pd.DataFrame(
        prediction_records
    )

    metrics = (
        metrics
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
    # Select best XGBoost configuration independently
    # for each forecast horizon.
    # --------------------------------------------------------

    best = (
        metrics
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

    best.to_csv(
        BEST_CONFIG_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP XGBOOST VALIDATION MODELS")
    print("=" * 70)

    for horizon in [
        1,
        2,
        4,
    ]:

        print(
            f"\nHORIZON {horizon} WEEK(S)"
        )

        top = (
            metrics[
                metrics[
                    "horizon_weeks"
                ]
                == horizon
            ]
            .head(10)
        )

        print(
            top[
                [
                    "feature_set",
                    "config_id",
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

    print("\n" + "=" * 70)
    print("BEST XGBOOST CONFIGURATION BY HORIZON")
    print("=" * 70)

    print(
        best[
            [
                "horizon_weeks",
                "feature_set",
                "config_id",
                "MAE",
                "RMSE",
                "MASE",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_lambda",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    print(
        f"\nMetrics saved to:\n"
        f"{METRICS_FILE}"
    )

    print(
        f"\nPredictions saved to:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"\nBest configurations saved to:\n"
        f"{BEST_CONFIG_FILE}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe 2026 test set was NOT evaluated."
    )


if __name__ == "__main__":
    main()