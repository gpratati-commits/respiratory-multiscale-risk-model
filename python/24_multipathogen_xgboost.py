"""
24_multipathogen_xgboost.py

Validation-stage XGBoost modelling for the multi-pathogen
ARI forecasting analysis.

Training:
    2024 targets

Validation / model selection:
    2025 targets

Held-out test:
    2026 is NOT evaluated here.

Feature-set comparison:
    1. ARI-only
    2. ARI + influenza
    3. ARI + COVID
    4. ARI + influenza + COVID

Forecast horizons:
    1, 2 and 4 weeks ahead.

Primary model-selection metric:
    MAE

Additional metrics:
    RMSE
    MASE
"""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multipathogen_forecasting_dataset.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_validation_metrics.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_validation_predictions.csv"
)

BEST_BY_FEATURE_SET_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_best_by_feature_set.csv"
)

BEST_OVERALL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_best_overall.csv"
)


# ============================================================
# SETTINGS
# ============================================================

HORIZONS = [1, 2, 4]


FEATURE_SETS = {

    "ari_only": [
        "ari_current",
        "ari_recent_4week_mean",
        "season_sin",
        "season_cos",
    ],

    "ari_plus_influenza": [
        "ari_current",
        "ari_recent_4week_mean",
        "season_sin",
        "season_cos",
        "influenza_lag1",
        "influenza_recent_4week_mean",
    ],

    "ari_plus_covid": [
        "ari_current",
        "ari_recent_4week_mean",
        "season_sin",
        "season_cos",
        "covid_lag1",
        "covid_recent_4week_mean",
    ],

    "ari_plus_influenza_covid": [
        "ari_current",
        "ari_recent_4week_mean",
        "season_sin",
        "season_cos",
        "influenza_lag1",
        "influenza_recent_4week_mean",
        "covid_lag1",
        "covid_recent_4week_mean",
    ],
}


# Small, regularized grid because the 2024 training set
# contains relatively few observations.

CONFIGURATIONS = [

    {
        "config_id": "A",
        "n_estimators": 50,
        "max_depth": 1,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 5.0,
    },

    {
        "config_id": "B",
        "n_estimators": 100,
        "max_depth": 1,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 5.0,
    },

    {
        "config_id": "C",
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 10.0,
    },

    {
        "config_id": "D",
        "n_estimators": 150,
        "max_depth": 1,
        "learning_rate": 0.02,
        "min_child_weight": 5,
        "subsample": 1.00,
        "colsample_bytree": 1.00,
        "reg_lambda": 10.0,
    },
]


# ============================================================
# METRICS
# ============================================================

def mae(y_true, y_pred):

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def rmse(y_true, y_pred):

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def mase(
    y_true,
    y_pred,
    training_target,
):

    training_target = np.asarray(
        training_target,
        dtype=float,
    )

    denominator = np.mean(
        np.abs(
            np.diff(
                training_target
            )
        )
    )

    if (
        denominator == 0
        or not np.isfinite(denominator)
    ):
        return np.nan

    return (
        mae(
            y_true,
            y_pred,
        )
        / denominator
    )


# ============================================================
# FIT ONE MODEL
# ============================================================

def fit_candidate(
    train,
    validation,
    horizon,
    feature_set_name,
    features,
    configuration,
):

    x_train = (
        train[features]
        .astype(float)
    )

    y_train = (
        train["target"]
        .astype(float)
        .to_numpy()
    )

    x_validation = (
        validation[features]
        .astype(float)
    )

    y_validation = (
        validation["target"]
        .astype(float)
        .to_numpy()
    )

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=(
            configuration[
                "n_estimators"
            ]
        ),

        max_depth=(
            configuration[
                "max_depth"
            ]
        ),

        learning_rate=(
            configuration[
                "learning_rate"
            ]
        ),

        min_child_weight=(
            configuration[
                "min_child_weight"
            ]
        ),

        subsample=(
            configuration[
                "subsample"
            ]
        ),

        colsample_bytree=(
            configuration[
                "colsample_bytree"
            ]
        ),

        reg_lambda=(
            configuration[
                "reg_lambda"
            ]
        ),

        random_state=42,

        n_jobs=2,

        verbosity=0,
    )

    model.fit(
        x_train,
        y_train,
    )

    prediction = model.predict(
        x_validation
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    prediction = np.clip(
        prediction,
        0,
        None,
    )

    metrics = {

        "horizon_weeks":
            horizon,

        "feature_set":
            feature_set_name,

        "config_id":
            configuration[
                "config_id"
            ],

        "n_train":
            len(train),

        "n_validation":
            len(validation),

        "MAE":
            mae(
                y_validation,
                prediction,
            ),

        "RMSE":
            rmse(
                y_validation,
                prediction,
            ),

        "MASE":
            mase(
                y_validation,
                prediction,
                y_train,
            ),

        "n_estimators":
            configuration[
                "n_estimators"
            ],

        "max_depth":
            configuration[
                "max_depth"
            ],

        "learning_rate":
            configuration[
                "learning_rate"
            ],

        "min_child_weight":
            configuration[
                "min_child_weight"
            ],

        "subsample":
            configuration[
                "subsample"
            ],

        "colsample_bytree":
            configuration[
                "colsample_bytree"
            ],

        "reg_lambda":
            configuration[
                "reg_lambda"
            ],
    }

    predictions = (
        validation[
            [
                "year",
                "epi_week",
                "target_year",
                "target_epi_week",
                "horizon_weeks",
                "target",
            ]
        ]
        .copy()
    )

    predictions[
        "feature_set"
    ] = feature_set_name

    predictions[
        "config_id"
    ] = configuration[
        "config_id"
    ]

    predictions[
        "prediction"
    ] = prediction

    predictions[
        "error"
    ] = (
        predictions[
            "target"
        ]
        -
        predictions[
            "prediction"
        ]
    )

    return (
        metrics,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("MULTI-PATHOGEN XGBOOST VALIDATION")
    print("=" * 76)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required = [
        "split",
        "horizon_weeks",
        "target",
        "target_year",
        "target_epi_week",
    ]

    for features in (
        FEATURE_SETS.values()
    ):

        required.extend(
            features
        )

    required = list(
        dict.fromkeys(
            required
        )
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            f"{missing}"
        )

    # --------------------------------------------------------
    # PROTECT 2026 TEST DATA
    # --------------------------------------------------------

    development = df[
        df["split"].isin(
            [
                "train",
                "validation",
            ]
        )
    ].copy()

    print(
        "\nDevelopment rows:",
        len(development),
    )

    print(
        "2026 test rows evaluated: 0"
    )

    metric_rows = []
    prediction_frames = []

    # --------------------------------------------------------
    # HORIZON LOOP
    # --------------------------------------------------------

    for horizon in HORIZONS:

        print(
            "\n" + "=" * 76
        )

        if horizon == 1:
            print(
                "HORIZON: 1 WEEK"
            )
        else:
            print(
                f"HORIZON: "
                f"{horizon} WEEKS"
            )

        print(
            "=" * 76
        )

        horizon_data = (
            development[
                development[
                    "horizon_weeks"
                ]
                == horizon
            ]
            .copy()
        )

        train = (
            horizon_data[
                horizon_data[
                    "split"
                ]
                == "train"
            ]
            .sort_values(
                [
                    "target_year",
                    "target_epi_week",
                ]
            )
            .copy()
        )

        validation = (
            horizon_data[
                horizon_data[
                    "split"
                ]
                == "validation"
            ]
            .sort_values(
                [
                    "target_year",
                    "target_epi_week",
                ]
            )
            .copy()
        )

        if train.empty:

            raise RuntimeError(
                f"No training data "
                f"for horizon {horizon}."
            )

        if validation.empty:

            raise RuntimeError(
                f"No validation data "
                f"for horizon {horizon}."
            )

        print(
            f"Training observations: "
            f"{len(train)}"
        )

        print(
            f"Validation observations: "
            f"{len(validation)}"
        )

        # ----------------------------------------------------
        # FEATURE SET LOOP
        # ----------------------------------------------------

        for (
            feature_set_name,
            features,
        ) in FEATURE_SETS.items():

            # ------------------------------------------------
            # CONFIGURATION LOOP
            # ------------------------------------------------

            for configuration in (
                CONFIGURATIONS
            ):

                print(
                    "  "
                    f"{feature_set_name} "
                    f"/ config "
                    f"{configuration['config_id']}"
                )

                try:

                    (
                        metrics,
                        predictions,
                    ) = fit_candidate(
                        train=train,
                        validation=validation,
                        horizon=horizon,
                        feature_set_name=(
                            feature_set_name
                        ),
                        features=features,
                        configuration=(
                            configuration
                        ),
                    )

                    metric_rows.append(
                        metrics
                    )

                    prediction_frames.append(
                        predictions
                    )

                except Exception as exc:

                    print(
                        "    FAILED: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        metric_rows
    )

    if metrics_df.empty:

        raise RuntimeError(
            "No XGBoost models completed."
        )

    metrics_df = (
        metrics_df
        .sort_values(
            [
                "horizon_weeks",
                "feature_set",
                "MAE",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # BEST CONFIGURATION WITHIN EACH FEATURE SET
    # --------------------------------------------------------

    best_by_feature_set = (
        metrics_df
        .sort_values(
            [
                "horizon_weeks",
                "feature_set",
                "MAE",
            ]
        )
        .groupby(
            [
                "horizon_weeks",
                "feature_set",
            ],
            as_index=False,
        )
        .first()
    )

    print(
        "\n" + "=" * 76
    )

    print(
        "BEST XGBOOST CONFIGURATION "
        "WITHIN EACH FEATURE SET"
    )

    print(
        "=" * 76
    )

    print(
        best_by_feature_set[
            [
                "horizon_weeks",
                "feature_set",
                "config_id",
                "MAE",
                "RMSE",
                "MASE",
            ]
        ]
        .sort_values(
            [
                "horizon_weeks",
                "MAE",
            ]
        )
        .round(3)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST OVERALL MODEL BY HORIZON
    # --------------------------------------------------------

    best_overall = (
        best_by_feature_set
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
        "\n" + "=" * 76
    )

    print(
        "BEST XGBOOST MODEL BY HORIZON"
    )

    print(
        "=" * 76
    )

    print(
        best_overall[
            [
                "horizon_weeks",
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

    # --------------------------------------------------------
    # PATHOGEN INCREMENTAL VALUE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 76
    )

    print(
        "PATHOGEN FEATURE-SET COMPARISON"
    )

    print(
        "=" * 76
    )

    for horizon in HORIZONS:

        comparison = (
            best_by_feature_set[
                best_by_feature_set[
                    "horizon_weeks"
                ]
                == horizon
            ]
            .sort_values("MAE")
        )

        print(
            f"\nHorizon {horizon}:"
        )

        print(
            comparison[
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

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    best_by_feature_set.to_csv(
        BEST_BY_FEATURE_SET_FILE,
        index=False,
    )

    best_overall.to_csv(
        BEST_OVERALL_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 76
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 76
    )

    print(
        f"\nAll validation metrics:\n"
        f"{METRICS_FILE}"
    )

    print(
        f"\nValidation predictions:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"\nBest configuration "
        f"within each feature set:\n"
        f"{BEST_BY_FEATURE_SET_FILE}"
    )

    print(
        f"\nBest overall XGBoost "
        f"models:\n"
        f"{BEST_OVERALL_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The 2026 held-out test set "
        "was NOT evaluated."
    )


if __name__ == "__main__":
    main()