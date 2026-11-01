"""
26_final_multipathogen_test_evaluation.py

Final held-out evaluation of the frozen multi-pathogen
forecasting models.

Model specifications were selected using validation data
before this script is run.

Development data:
    targets in 2024 + 2025

Held-out test:
    targets in 2026

IMPORTANT
---------
The selected feature set/model family/configuration is read
from:

outputs/tables/multipathogen_final_validation_selection.csv

The 2026 test results MUST NOT be used to change the frozen
model specification.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multipathogen_forecasting_dataset.csv"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_final_validation_selection.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_final_test_metrics.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_final_test_predictions.csv"
)


# ============================================================
# FEATURE SETS
# ============================================================

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


# ============================================================
# XGBOOST CONFIGURATIONS
# These must match Step 24 exactly.
# ============================================================

XGB_CONFIGS = {

    "A": {
        "n_estimators": 50,
        "max_depth": 1,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 5.0,
    },

    "B": {
        "n_estimators": 100,
        "max_depth": 1,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 5.0,
    },

    "C": {
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 10.0,
    },

    "D": {
        "n_estimators": 150,
        "max_depth": 1,
        "learning_rate": 0.02,
        "min_child_weight": 5,
        "subsample": 1.00,
        "colsample_bytree": 1.00,
        "reg_lambda": 10.0,
    },
}


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
    development_target,
):

    development_target = np.asarray(
        development_target,
        dtype=float,
    )

    denominator = np.mean(
        np.abs(
            np.diff(
                development_target
            )
        )
    )

    if (
        not np.isfinite(denominator)
        or denominator == 0
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
# NEGATIVE BINOMIAL HELPERS
# ============================================================

def estimate_alpha(y):

    y = np.asarray(
        y,
        dtype=float,
    )

    mean_y = np.mean(y)

    var_y = np.var(
        y,
        ddof=1,
    )

    if mean_y <= 0:
        return 1e-6

    alpha = (
        var_y - mean_y
    ) / (
        mean_y ** 2
    )

    return max(
        float(alpha),
        1e-6,
    )


def fit_negative_binomial(
    development,
    test,
    features,
):

    x_dev = (
        development[features]
        .astype(float)
        .copy()
    )

    x_test = (
        test[features]
        .astype(float)
        .copy()
    )

    y_dev = (
        development["target"]
        .astype(float)
        .to_numpy()
    )

    means = x_dev.mean()

    stds = (
        x_dev
        .std(ddof=0)
        .replace(0, 1.0)
    )

    x_dev = (
        x_dev - means
    ) / stds

    x_test = (
        x_test - means
    ) / stds

    x_dev = sm.add_constant(
        x_dev,
        has_constant="add",
    )

    x_test = sm.add_constant(
        x_test,
        has_constant="add",
    )

    alpha = estimate_alpha(
        y_dev
    )

    family = (
        sm.families.NegativeBinomial(
            alpha=alpha
        )
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        model = sm.GLM(
            y_dev,
            x_dev,
            family=family,
        )

        result = model.fit(
            maxiter=200,
            disp=0,
        )

    predictions = np.asarray(
        result.predict(
            x_test
        ),
        dtype=float,
    )

    predictions = np.clip(
        predictions,
        0,
        None,
    )

    return predictions


# ============================================================
# XGBOOST FIT
# ============================================================

def fit_xgboost(
    development,
    test,
    features,
    config_id,
):

    if config_id not in XGB_CONFIGS:

        raise ValueError(
            f"Unknown XGBoost configuration: "
            f"{config_id}"
        )

    config = XGB_CONFIGS[
        config_id
    ]

    x_dev = (
        development[features]
        .astype(float)
    )

    y_dev = (
        development["target"]
        .astype(float)
        .to_numpy()
    )

    x_test = (
        test[features]
        .astype(float)
    )

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=(
            config["n_estimators"]
        ),

        max_depth=(
            config["max_depth"]
        ),

        learning_rate=(
            config["learning_rate"]
        ),

        min_child_weight=(
            config["min_child_weight"]
        ),

        subsample=(
            config["subsample"]
        ),

        colsample_bytree=(
            config["colsample_bytree"]
        ),

        reg_lambda=(
            config["reg_lambda"]
        ),

        random_state=42,

        n_jobs=2,

        verbosity=0,
    )

    model.fit(
        x_dev,
        y_dev,
    )

    predictions = np.asarray(
        model.predict(
            x_test
        ),
        dtype=float,
    )

    return np.clip(
        predictions,
        0,
        None,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("FINAL MULTI-PATHOGEN HELD-OUT TEST EVALUATION")
    print("=" * 78)

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Forecasting dataset not found:\n"
            f"{DATA_FILE}"
        )

    if not SELECTION_FILE.exists():

        raise FileNotFoundError(
            f"Frozen model selection not found:\n"
            f"{SELECTION_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    selection = pd.read_csv(
        SELECTION_FILE
    )

    print(
        "\nFrozen specifications:"
    )

    print(
        selection[
            [
                "horizon_weeks",
                "model_family",
                "feature_set",
                "configuration",
            ]
        ]
        .sort_values("horizon_weeks")
        .to_string(index=False)
    )

    metric_rows = []
    prediction_frames = []

    # --------------------------------------------------------
    # EACH FROZEN HORIZON
    # --------------------------------------------------------

    for _, selected in (
        selection
        .sort_values("horizon_weeks")
        .iterrows()
    ):

        horizon = int(
            selected[
                "horizon_weeks"
            ]
        )

        model_family = str(
            selected[
                "model_family"
            ]
        )

        feature_set = str(
            selected[
                "feature_set"
            ]
        )

        configuration = str(
            selected[
                "configuration"
            ]
        )

        if feature_set not in FEATURE_SETS:

            raise ValueError(
                f"Unknown feature set: "
                f"{feature_set}"
            )

        features = FEATURE_SETS[
            feature_set
        ]

        horizon_df = df[
            df["horizon_weeks"]
            == horizon
        ].copy()

        # --------------------------------------------
        # Development = target years 2024 + 2025
        # --------------------------------------------

        development = (
            horizon_df[
                horizon_df["split"]
                .isin(
                    [
                        "train",
                        "validation",
                    ]
                )
            ]
            .sort_values(
                [
                    "target_year",
                    "target_epi_week",
                ]
            )
            .copy()
        )

        # --------------------------------------------
        # Held-out test = target year 2026
        # --------------------------------------------

        test = (
            horizon_df[
                horizon_df["split"]
                == "test"
            ]
            .sort_values(
                [
                    "target_year",
                    "target_epi_week",
                ]
            )
            .copy()
        )

        if development.empty:

            raise RuntimeError(
                f"No development rows for "
                f"horizon {horizon}."
            )

        if test.empty:

            raise RuntimeError(
                f"No test rows for "
                f"horizon {horizon}."
            )

        print(
            "\n" + "=" * 78
        )

        print(
            f"HORIZON {horizon}"
        )

        print(
            "=" * 78
        )

        print(
            f"Model family: {model_family}"
        )

        print(
            f"Feature set: {feature_set}"
        )

        print(
            f"Configuration: {configuration}"
        )

        print(
            f"Development observations: "
            f"{len(development)}"
        )

        print(
            f"Test observations: "
            f"{len(test)}"
        )

        # --------------------------------------------
        # FIT FROZEN MODEL
        # --------------------------------------------

        if (
            model_family
            == "negative_binomial"
        ):

            predictions = (
                fit_negative_binomial(
                    development=development,
                    test=test,
                    features=features,
                )
            )

        elif (
            model_family
            == "xgboost"
        ):

            predictions = (
                fit_xgboost(
                    development=development,
                    test=test,
                    features=features,
                    config_id=configuration,
                )
            )

        else:

            raise ValueError(
                f"Unknown model family: "
                f"{model_family}"
            )

        y_test = (
            test["target"]
            .astype(float)
            .to_numpy()
        )

        y_dev = (
            development["target"]
            .astype(float)
            .to_numpy()
        )

        horizon_mae = mae(
            y_test,
            predictions,
        )

        horizon_rmse = rmse(
            y_test,
            predictions,
        )

        horizon_mase = mase(
            y_test,
            predictions,
            y_dev,
        )

        metric_rows.append(
            {
                "horizon_weeks":
                    horizon,

                "model_family":
                    model_family,

                "feature_set":
                    feature_set,

                "configuration":
                    configuration,

                "n_development":
                    len(development),

                "n_test":
                    len(test),

                "MAE":
                    horizon_mae,

                "RMSE":
                    horizon_rmse,

                "MASE":
                    horizon_mase,
            }
        )

        prediction_table = (
            test[
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

        prediction_table[
            "model_family"
        ] = model_family

        prediction_table[
            "feature_set"
        ] = feature_set

        prediction_table[
            "configuration"
        ] = configuration

        prediction_table[
            "prediction"
        ] = predictions

        prediction_table[
            "error"
        ] = (
            prediction_table[
                "target"
            ]
            -
            prediction_table[
                "prediction"
            ]
        )

        prediction_frames.append(
            prediction_table
        )

        print(
            f"\nMAE:  "
            f"{horizon_mae:.3f}"
        )

        print(
            f"RMSE: "
            f"{horizon_rmse:.3f}"
        )

        print(
            f"MASE: "
            f"{horizon_mase:.3f}"
        )

    # --------------------------------------------------------
    # FINAL TABLE
    # --------------------------------------------------------

    metrics_df = (
        pd.DataFrame(
            metric_rows
        )
        .sort_values(
            "horizon_weeks"
        )
        .reset_index(drop=True)
    )

    predictions_df = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "FINAL HELD-OUT TEST RESULTS — 2026"
    )

    print(
        "=" * 78
    )

    print(
        metrics_df[
            [
                "horizon_weeks",
                "model_family",
                "feature_set",
                "configuration",
                "n_test",
                "MAE",
                "RMSE",
                "MASE",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

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

    print(
        "\n" + "=" * 78
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 78
    )

    print(
        f"\nFinal test metrics:\n"
        f"{METRICS_FILE}"
    )

    print(
        f"\nFinal test predictions:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are the final frozen-model "
        "2026 held-out results."
    )

    print(
        "Do NOT change the model specification "
        "based on these results."
    )


if __name__ == "__main__":
    main()