"""
23_multipathogen_negative_binomial.py

Validation-stage Negative Binomial modelling for the
multi-pathogen ARI forecasting analysis.

IMPORTANT
---------
Model fitting:
    2024 targets only

Model selection:
    2025 validation targets only

Held-out test:
    2026 is NOT evaluated in this script.

Models compared
---------------
1. ARI-only
2. ARI + influenza
3. ARI + COVID
4. ARI + influenza + COVID

Forecast horizons
-----------------
1, 2 and 4 weeks ahead.

Primary model-selection metric:
    MAE

Additional metrics:
    RMSE
    MASE
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    / "multipathogen_nb_validation_metrics.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_nb_validation_predictions.csv"
)

BEST_MODELS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_nb_best_models.csv"
)

COEFFICIENTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_nb_coefficients.csv"
)


# ============================================================
# SETTINGS
# ============================================================

HORIZONS = [1, 2, 4]


# Compact predictor sets are deliberate because the
# 2024 training period is relatively short.

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
    """
    Mean Absolute Scaled Error.

    Scale denominator is calculated from the chronological
    training target series only.
    """

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
# NEGATIVE BINOMIAL DISPERSION
# ============================================================

def estimate_alpha(y):
    """
    Estimate a simple Negative Binomial dispersion parameter
    from the training data using:

        Var(Y) = mean + alpha * mean^2

    alpha is estimated using TRAINING DATA ONLY.
    """

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

    # Prevent invalid/negative dispersion.
    alpha = max(
        float(alpha),
        1e-6,
    )

    return alpha


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_predictors(
    train,
    validation,
    features,
):
    """
    Standardize predictors using TRAINING statistics only.

    Validation means/SDs are never used for scaling.
    """

    x_train = (
        train[features]
        .astype(float)
        .copy()
    )

    x_validation = (
        validation[features]
        .astype(float)
        .copy()
    )

    means = x_train.mean()

    stds = x_train.std(
        ddof=0
    )

    # Avoid division by zero for constant predictors.
    stds = stds.replace(
        0,
        1.0,
    )

    x_train = (
        x_train - means
    ) / stds

    x_validation = (
        x_validation - means
    ) / stds

    x_train = sm.add_constant(
        x_train,
        has_constant="add",
    )

    x_validation = sm.add_constant(
        x_validation,
        has_constant="add",
    )

    return (
        x_train,
        x_validation,
    )


# ============================================================
# MODEL FIT
# ============================================================

def fit_candidate_model(
    train,
    validation,
    feature_set_name,
    features,
    horizon,
):
    """
    Fit one Negative Binomial candidate model and
    evaluate it on 2025 validation data.
    """

    y_train = (
        train["target"]
        .astype(float)
        .to_numpy()
    )

    y_validation = (
        validation["target"]
        .astype(float)
        .to_numpy()
    )

    (
        x_train,
        x_validation,
    ) = standardize_predictors(
        train=train,
        validation=validation,
        features=features,
    )

    alpha = estimate_alpha(
        y_train
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
            y_train,
            x_train,
            family=family,
        )

        result = model.fit(
            maxiter=200,
            disp=0,
        )

    prediction = np.asarray(
        result.predict(
            x_validation
        ),
        dtype=float,
    )

    # Negative predictions should not occur under log-link,
    # but clipping is a final safety check.
    prediction = np.clip(
        prediction,
        0,
        None,
    )

    converged = bool(
        getattr(
            result,
            "converged",
            True,
        )
    )

    metrics = {

        "horizon_weeks":
            horizon,

        "feature_set":
            feature_set_name,

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

        "alpha":
            alpha,

        "converged":
            converged,
    }

    prediction_table = (
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

    prediction_table[
        "feature_set"
    ] = feature_set_name

    prediction_table[
        "prediction"
    ] = prediction

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

    coefficient_rows = []

    for parameter_name, value in (
        result.params.items()
    ):

        coefficient_rows.append(
            {
                "horizon_weeks":
                    horizon,

                "feature_set":
                    feature_set_name,

                "parameter":
                    parameter_name,

                "coefficient":
                    float(value),
            }
        )

    return (
        metrics,
        prediction_table,
        coefficient_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("MULTI-PATHOGEN NEGATIVE BINOMIAL VALIDATION")
    print("=" * 76)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "split",
        "horizon_weeks",
        "target",
    ]

    for features in FEATURE_SETS.values():
        required_columns.extend(features)

    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # DROP TEST DATA FROM THIS ANALYSIS
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
        "\nRows available for development:",
        len(development),
    )

    print(
        "2026 test rows evaluated: 0"
    )

    # --------------------------------------------------------
    # MODEL LOOP
    # --------------------------------------------------------

    metric_rows = []
    prediction_frames = []
    coefficient_rows = []

    for horizon in HORIZONS:

        print(
            "\n" + "=" * 76
        )

        print(
            f"HORIZON: {horizon} WEEK"
            if horizon == 1
            else f"HORIZON: {horizon} WEEKS"
        )

        print(
            "=" * 76
        )

        horizon_data = development[
            development[
                "horizon_weeks"
            ]
            == horizon
        ].copy()

        train = (
            horizon_data[
                horizon_data["split"]
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
                horizon_data["split"]
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
                f"No training observations "
                f"for horizon {horizon}."
            )

        if validation.empty:
            raise RuntimeError(
                f"No validation observations "
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

        for (
            feature_set_name,
            features,
        ) in FEATURE_SETS.items():

            print(
                f"  Fitting: "
                f"{feature_set_name}"
            )

            try:

                (
                    metrics,
                    predictions,
                    coefficients,
                ) = fit_candidate_model(
                    train=train,
                    validation=validation,
                    feature_set_name=(
                        feature_set_name
                    ),
                    features=features,
                    horizon=horizon,
                )

                metric_rows.append(
                    metrics
                )

                prediction_frames.append(
                    predictions
                )

                coefficient_rows.extend(
                    coefficients
                )

            except Exception as exc:

                print(
                    f"    FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                metric_rows.append(
                    {
                        "horizon_weeks":
                            horizon,

                        "feature_set":
                            feature_set_name,

                        "n_train":
                            len(train),

                        "n_validation":
                            len(validation),

                        "MAE":
                            np.nan,

                        "RMSE":
                            np.nan,

                        "MASE":
                            np.nan,

                        "alpha":
                            np.nan,

                        "converged":
                            False,
                    }
                )

    # --------------------------------------------------------
    # METRICS TABLE
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df = (
        metrics_df
        .sort_values(
            [
                "horizon_weeks",
                "MAE",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n" + "=" * 76
    )

    print(
        "VALIDATION RESULTS — NEGATIVE BINOMIAL"
    )

    print(
        "=" * 76
    )

    print(
        metrics_df[
            [
                "horizon_weeks",
                "feature_set",
                "n_train",
                "n_validation",
                "MAE",
                "RMSE",
                "MASE",
                "alpha",
                "converged",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST MODEL BY HORIZON
    # --------------------------------------------------------

    valid_metrics = metrics_df[
        metrics_df["MAE"].notna()
    ].copy()

    if valid_metrics.empty:

        raise RuntimeError(
            "All Negative Binomial "
            "candidate models failed."
        )

    best_models = (
        valid_metrics
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
        "BEST NB MODEL BY HORIZON"
    )

    print(
        "=" * 76
    )

    print(
        best_models[
            [
                "horizon_weeks",
                "feature_set",
                "MAE",
                "RMSE",
                "MASE",
                "converged",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    coefficients_df = pd.DataFrame(
        coefficient_rows
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

    best_models.to_csv(
        BEST_MODELS_FILE,
        index=False,
    )

    coefficients_df.to_csv(
        COEFFICIENTS_FILE,
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
        f"\nValidation metrics:\n"
        f"{METRICS_FILE}"
    )

    print(
        f"\nValidation predictions:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"\nBest NB specifications:\n"
        f"{BEST_MODELS_FILE}"
    )

    print(
        f"\nModel coefficients:\n"
        f"{COEFFICIENTS_FILE}"
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