"""
27_multipathogen_sensitivity_analysis.py

Sensitivity / robustness analysis for the final frozen
multi-pathogen ARI forecasting models.

PURPOSE
-------
Assess whether the final conclusions are sensitive to how
source-reported COVID-19 positivity values "<1%" are represented.

Primary analysis:
    <1% -> 0.5

Sensitivity scenarios:
    <1% -> 0.0
    <1% -> 0.5   [PRIMARY]
    <1% -> 1.0

IMPORTANT
---------
This script does NOT re-select models.

The model family, feature set, and XGBoost configuration for
each forecast horizon remain exactly those frozen during
validation before the 2026 held-out evaluation.

Sensitivity results must NOT be used to replace the primary
model specifications.
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

WEEKLY_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multipathogen_weekly_dataset.csv"
)

FORECAST_TEMPLATE_FILE = (
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

PRIMARY_TEST_METRICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_final_test_metrics.csv"
)

SENSITIVITY_METRICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_covid_censoring_sensitivity.csv"
)

SENSITIVITY_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_covid_censoring_sensitivity_predictions.csv"
)

PRIMARY_REPRODUCTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_sensitivity_primary_reproduction_qc.csv"
)


# ============================================================
# SETTINGS
# ============================================================

COVID_CENSOR_SCENARIOS = {
    "covid_lt1_as_0": 0.0,
    "primary_covid_lt1_as_0_5": 0.5,
    "covid_lt1_as_1": 1.0,
}

PRIMARY_SCENARIO = "primary_covid_lt1_as_0_5"

# Primary reconstruction should reproduce Step 26 to within
# one thousandth for MAE / RMSE / MASE.
REPRODUCTION_TOLERANCE = 1e-3


# ============================================================
# FEATURE SETS
# Must match Steps 23-26 exactly.
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
# Must match Step 24 / Step 26 exactly.
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
# COVID-DERIVED FEATURES
# ============================================================

COVID_DERIVED_FEATURES = [
    "covid_lag1",
    "covid_lag2",
    "covid_lag4",
    "covid_recent_2week_mean",
    "covid_recent_4week_mean",
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
# BOOLEAN HANDLING
# ============================================================

def parse_boolean_series(series):
    """
    Convert a CSV boolean column safely.

    This avoids the dangerous behaviour where:
        bool("False") == True
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    if series.isna().any():
        raise ValueError(
            "covid_censored contains missing values."
        )

    normalised = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    invalid = normalised[
        ~normalised.isin(
            mapping.keys()
        )
    ].unique()

    if len(invalid) > 0:
        raise ValueError(
            "Unexpected covid_censored values: "
            f"{invalid}"
        )

    return normalised.map(mapping).astype(bool)


# ============================================================
# BUILD COVID FEATURES FOR ONE SCENARIO
# ============================================================

def build_covid_features(
    weekly_source,
    censored_value,
):

    weekly = (
        weekly_source
        .copy()
        .sort_values(
            [
                "year",
                "epi_week",
            ]
        )
        .reset_index(drop=True)
    )

    required = [
        "year",
        "epi_week",
        "covid_positivity",
        "covid_censored",
    ]

    missing = [
        column
        for column in required
        if column not in weekly.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in weekly dataset: "
            f"{missing}"
        )

    weekly["covid_censored"] = (
        parse_boolean_series(
            weekly["covid_censored"]
        )
    )

    weekly["covid_positivity"] = (
        pd.to_numeric(
            weekly["covid_positivity"],
            errors="raise",
        )
    )

    censored_mask = (
        weekly["covid_censored"]
        == True
    )

    weekly.loc[
        censored_mask,
        "covid_positivity"
    ] = float(censored_value)

    # --------------------------------------------------------
    # EXACT STEP 21 DEFINITIONS
    #
    # Lag 1 = previous observed week
    # Lag 2 = two weeks previously
    # Lag 4 = four weeks previously
    #
    # Rolling means EXCLUDE the current week.
    # --------------------------------------------------------

    for lag in [1, 2, 4]:

        weekly[
            f"covid_lag{lag}"
        ] = (
            weekly[
                "covid_positivity"
            ]
            .shift(lag)
        )

    weekly[
        "covid_recent_2week_mean"
    ] = (
        weekly[
            "covid_positivity"
        ]
        .shift(1)
        .rolling(
            window=2,
            min_periods=2,
        )
        .mean()
    )

    weekly[
        "covid_recent_4week_mean"
    ] = (
        weekly[
            "covid_positivity"
        ]
        .shift(1)
        .rolling(
            window=4,
            min_periods=4,
        )
        .mean()
    )

    return weekly[
        [
            "year",
            "epi_week",
            "covid_positivity",
            *COVID_DERIVED_FEATURES,
        ]
    ].copy()


# ============================================================
# CREATE SCENARIO FORECASTING DATASET
# ============================================================

def build_scenario_forecast_dataset(
    weekly_source,
    forecast_template,
    censored_value,
):

    covid_features = (
        build_covid_features(
            weekly_source=weekly_source,
            censored_value=censored_value,
        )
    )

    scenario = forecast_template.copy()

    # Preserve exact Step 22 row order.
    scenario["_original_row_order"] = (
        np.arange(
            len(scenario)
        )
    )

    columns_to_replace = [
        "covid_positivity",
        *COVID_DERIVED_FEATURES,
    ]

    scenario = scenario.drop(
        columns=[
            column
            for column in columns_to_replace
            if column in scenario.columns
        ],
        errors="ignore",
    )

    scenario = scenario.merge(
        covid_features,
        on=[
            "year",
            "epi_week",
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    scenario = (
        scenario
        .sort_values(
            "_original_row_order"
        )
        .drop(
            columns=[
                "_original_row_order"
            ]
        )
        .reset_index(drop=True)
    )

    return scenario


# ============================================================
# NEGATIVE BINOMIAL
# ============================================================

def estimate_alpha(y):

    y = np.asarray(
        y,
        dtype=float,
    )

    mean_y = np.mean(y)

    variance_y = np.var(
        y,
        ddof=1,
    )

    if mean_y <= 0:
        return 1e-6

    alpha = (
        variance_y - mean_y
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
        .replace(
            0,
            1.0,
        )
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

    prediction = np.asarray(
        result.predict(
            x_test
        ),
        dtype=float,
    )

    return np.clip(
        prediction,
        0,
        None,
    )


# ============================================================
# XGBOOST
# ============================================================

def fit_xgboost(
    development,
    test,
    features,
    configuration,
):

    configuration = str(
        configuration
    ).strip()

    if configuration not in XGB_CONFIGS:
        raise ValueError(
            "Unknown XGBoost configuration: "
            f"{configuration}"
        )

    config = XGB_CONFIGS[
        configuration
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
            config[
                "n_estimators"
            ]
        ),

        max_depth=(
            config[
                "max_depth"
            ]
        ),

        learning_rate=(
            config[
                "learning_rate"
            ]
        ),

        min_child_weight=(
            config[
                "min_child_weight"
            ]
        ),

        subsample=(
            config[
                "subsample"
            ]
        ),

        colsample_bytree=(
            config[
                "colsample_bytree"
            ]
        ),

        reg_lambda=(
            config[
                "reg_lambda"
            ]
        ),

        random_state=42,

        n_jobs=2,

        verbosity=0,
    )

    model.fit(
        x_dev,
        y_dev,
    )

    prediction = np.asarray(
        model.predict(
            x_test
        ),
        dtype=float,
    )

    return np.clip(
        prediction,
        0,
        None,
    )


# ============================================================
# FIT ONE FROZEN MODEL
# ============================================================

def evaluate_frozen_model(
    scenario_dataset,
    selected,
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
    ).strip()

    feature_set = str(
        selected[
            "feature_set"
        ]
    ).strip()

    configuration = str(
        selected[
            "configuration"
        ]
    ).strip()

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            "Unknown feature set: "
            f"{feature_set}"
        )

    features = FEATURE_SETS[
        feature_set
    ]

    horizon_df = (
        scenario_dataset[
            scenario_dataset[
                "horizon_weeks"
            ]
            == horizon
        ]
        .copy()
    )

    development = (
        horizon_df[
            horizon_df[
                "split"
            ].isin(
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

    test = (
        horizon_df[
            horizon_df[
                "split"
            ]
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

    required_for_model = (
        features
        + ["target"]
    )

    missing_dev = (
        development[
            required_for_model
        ]
        .isna()
        .any(axis=1)
    )

    missing_test = (
        test[
            required_for_model
        ]
        .isna()
        .any(axis=1)
    )

    if missing_dev.any():
        raise RuntimeError(
            f"Missing development predictors "
            f"for horizon {horizon}: "
            f"{int(missing_dev.sum())} rows."
        )

    if missing_test.any():
        raise RuntimeError(
            f"Missing test predictors "
            f"for horizon {horizon}: "
            f"{int(missing_test.sum())} rows."
        )

    if development.empty:
        raise RuntimeError(
            f"No development observations "
            f"for horizon {horizon}."
        )

    if test.empty:
        raise RuntimeError(
            f"No test observations "
            f"for horizon {horizon}."
        )

    if model_family == "negative_binomial":

        prediction = (
            fit_negative_binomial(
                development=development,
                test=test,
                features=features,
            )
        )

    elif model_family == "xgboost":

        prediction = (
            fit_xgboost(
                development=development,
                test=test,
                features=features,
                configuration=configuration,
            )
        )

    else:

        raise ValueError(
            "Unknown model family: "
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

    metrics = {

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
            mae(
                y_test,
                prediction,
            ),

        "RMSE":
            rmse(
                y_test,
                prediction,
            ),

        "MASE":
            mase(
                y_test,
                prediction,
                y_dev,
            ),
    }

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

    return (
        metrics,
        prediction_table,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print("MULTI-PATHOGEN COVID CENSORING SENSITIVITY ANALYSIS")
    print("=" * 82)

    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    required_files = [
        WEEKLY_DATA_FILE,
        FORECAST_TEMPLATE_FILE,
        SELECTION_FILE,
        PRIMARY_TEST_METRICS_FILE,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    weekly = pd.read_csv(
        WEEKLY_DATA_FILE
    )

    template = pd.read_csv(
        FORECAST_TEMPLATE_FILE
    )

    selection = pd.read_csv(
        SELECTION_FILE
    )

    primary_test_metrics = pd.read_csv(
        PRIMARY_TEST_METRICS_FILE
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    required_weekly_columns = [
        "year",
        "epi_week",
        "covid_positivity",
        "covid_censored",
    ]

    missing_weekly = [
        column
        for column
        in required_weekly_columns
        if column not in weekly.columns
    ]

    if missing_weekly:
        raise ValueError(
            "Weekly dataset is missing columns: "
            f"{missing_weekly}"
        )

    required_template_columns = [
        "year",
        "epi_week",
        "horizon_weeks",
        "target_year",
        "target_epi_week",
        "split",
        "target",
    ]

    missing_template = [
        column
        for column
        in required_template_columns
        if column not in template.columns
    ]

    if missing_template:
        raise ValueError(
            "Forecasting template is missing columns: "
            f"{missing_template}"
        )

    required_selection_columns = [
        "horizon_weeks",
        "model_family",
        "feature_set",
        "configuration",
    ]

    missing_selection = [
        column
        for column
        in required_selection_columns
        if column not in selection.columns
    ]

    if missing_selection:
        raise ValueError(
            "Frozen selection file is missing columns: "
            f"{missing_selection}"
        )

    selection = (
        selection
        .sort_values(
            "horizon_weeks"
        )
        .reset_index(drop=True)
    )

    print(
        "\nFrozen specifications:"
    )

    print(
        selection[
            required_selection_columns
        ]
        .to_string(
            index=False
        )
    )

    covid_censored = (
        parse_boolean_series(
            weekly[
                "covid_censored"
            ]
        )
    )

    print(
        "\nSource-reported COVID <1% "
        "observations:"
    )

    print(
        int(
            covid_censored.sum()
        )
    )

    # --------------------------------------------------------
    # PRIMARY FEATURE RECONSTRUCTION QC
    # --------------------------------------------------------

    print(
        "\n" + "=" * 82
    )

    print(
        "PRIMARY 0.5% COVID FEATURE RECONSTRUCTION QC"
    )

    print(
        "=" * 82
    )

    primary_scenario_dataset = (
        build_scenario_forecast_dataset(
            weekly_source=weekly,
            forecast_template=template,
            censored_value=0.5,
        )
    )

    common_covid_features = [
        column
        for column
        in COVID_DERIVED_FEATURES
        if column in template.columns
    ]

    if common_covid_features:

        for column in common_covid_features:

            original = pd.to_numeric(
                template[column],
                errors="coerce",
            ).to_numpy()

            reconstructed = pd.to_numeric(
                primary_scenario_dataset[
                    column
                ],
                errors="coerce",
            ).to_numpy()

            valid = (
                np.isfinite(original)
                &
                np.isfinite(reconstructed)
            )

            if valid.any():

                max_difference = float(
                    np.max(
                        np.abs(
                            original[valid]
                            - reconstructed[valid]
                        )
                    )
                )

            else:

                max_difference = np.nan

            print(
                f"{column}: "
                f"max absolute difference = "
                f"{max_difference}"
            )

    # --------------------------------------------------------
    # RUN ALL THREE SCENARIOS
    # --------------------------------------------------------

    metric_rows = []
    prediction_frames = []

    for (
        scenario_name,
        censored_value,
    ) in COVID_CENSOR_SCENARIOS.items():

        print(
            "\n" + "=" * 82
        )

        print(
            f"SCENARIO: {scenario_name}"
        )

        print(
            f"Represent COVID <1% as "
            f"{censored_value}%"
        )

        print(
            "=" * 82
        )

        scenario_dataset = (
            build_scenario_forecast_dataset(
                weekly_source=weekly,
                forecast_template=template,
                censored_value=censored_value,
            )
        )

        for _, selected in (
            selection.iterrows()
        ):

            (
                model_metrics,
                prediction_table,
            ) = evaluate_frozen_model(
                scenario_dataset=(
                    scenario_dataset
                ),
                selected=selected,
            )

            model_metrics[
                "scenario"
            ] = scenario_name

            model_metrics[
                "covid_censored_value"
            ] = float(
                censored_value
            )

            metric_rows.append(
                model_metrics
            )

            prediction_table[
                "scenario"
            ] = scenario_name

            prediction_table[
                "covid_censored_value"
            ] = float(
                censored_value
            )

            prediction_frames.append(
                prediction_table
            )

            print(
                "\n"
                f"Horizon: "
                f"{model_metrics['horizon_weeks']}"
            )

            print(
                f"Model: "
                f"{model_metrics['model_family']}"
            )

            print(
                f"Feature set: "
                f"{model_metrics['feature_set']}"
            )

            print(
                f"MAE: "
                f"{model_metrics['MAE']:.3f}"
            )

            print(
                f"RMSE: "
                f"{model_metrics['RMSE']:.3f}"
            )

            print(
                f"MASE: "
                f"{model_metrics['MASE']:.3f}"
            )

    metrics = pd.DataFrame(
        metric_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # PRIMARY SCENARIO MUST REPRODUCE STEP 26
    # --------------------------------------------------------

    print(
        "\n" + "=" * 82
    )

    print(
        "PRIMARY SCENARIO REPRODUCTION CHECK"
    )

    print(
        "=" * 82
    )

    primary_reproduction = (
        metrics[
            metrics[
                "scenario"
            ]
            == PRIMARY_SCENARIO
        ][
            [
                "horizon_weeks",
                "MAE",
                "RMSE",
                "MASE",
                "n_test",
            ]
        ]
        .merge(
            primary_test_metrics[
                [
                    "horizon_weeks",
                    "MAE",
                    "RMSE",
                    "MASE",
                    "n_test",
                ]
            ],
            on="horizon_weeks",
            how="inner",
            suffixes=(
                "_sensitivity",
                "_step26",
            ),
            validate="one_to_one",
        )
    )

    for metric in [
        "MAE",
        "RMSE",
        "MASE",
    ]:

        primary_reproduction[
            f"abs_diff_{metric}"
        ] = np.abs(
            primary_reproduction[
                f"{metric}_sensitivity"
            ]
            -
            primary_reproduction[
                f"{metric}_step26"
            ]
        )

    print(
        primary_reproduction[
            [
                "horizon_weeks",
                "MAE_sensitivity",
                "MAE_step26",
                "abs_diff_MAE",
                "RMSE_sensitivity",
                "RMSE_step26",
                "abs_diff_RMSE",
                "MASE_sensitivity",
                "MASE_step26",
                "abs_diff_MASE",
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    maximum_reproduction_difference = (
        primary_reproduction[
            [
                "abs_diff_MAE",
                "abs_diff_RMSE",
                "abs_diff_MASE",
            ]
        ]
        .to_numpy()
        .max()
    )

    if (
        maximum_reproduction_difference
        > REPRODUCTION_TOLERANCE
    ):

        raise RuntimeError(
            "\nPRIMARY REPRODUCTION FAILED.\n"
            "The 0.5% sensitivity scenario did "
            "not reproduce Step 26 closely enough.\n"
            f"Maximum difference = "
            f"{maximum_reproduction_difference:.6f}\n"
            "Do not interpret the sensitivity "
            "results until this is corrected."
        )

    print(
        "\nPRIMARY REPRODUCTION PASSED."
    )

    # --------------------------------------------------------
    # QC:
    # MODELS WITHOUT COVID MUST NOT CHANGE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 82
    )

    print(
        "NON-COVID MODEL INVARIANCE CHECK"
    )

    print(
        "=" * 82
    )

    for _, selected in (
        selection.iterrows()
    ):

        horizon = int(
            selected[
                "horizon_weeks"
            ]
        )

        feature_set = str(
            selected[
                "feature_set"
            ]
        )

        if "covid" in feature_set:
            continue

        subset = metrics[
            metrics[
                "horizon_weeks"
            ]
            == horizon
        ]

        for metric in [
            "MAE",
            "RMSE",
            "MASE",
        ]:

            spread = float(
                subset[metric].max()
                -
                subset[metric].min()
            )

            print(
                f"Horizon {horizon}, "
                f"{metric} scenario spread: "
                f"{spread:.12f}"
            )

            if spread > 1e-9:

                raise RuntimeError(
                    f"Unexpected change in "
                    f"non-COVID model at "
                    f"horizon {horizon}."
                )

    print(
        "\nNON-COVID MODEL INVARIANCE PASSED."
    )

    # --------------------------------------------------------
    # SENSITIVITY RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 82
    )

    print(
        "COVID CENSORING SENSITIVITY RESULTS"
    )

    print(
        "=" * 82
    )

    sensitivity_display = (
        metrics
        .sort_values(
            [
                "horizon_weeks",
                "covid_censored_value",
            ]
        )[
            [
                "scenario",
                "covid_censored_value",
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
    )

    print(
        sensitivity_display
        .round(3)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DIFFERENCE RELATIVE TO PRIMARY
    # --------------------------------------------------------

    primary_reference = (
        metrics[
            metrics[
                "scenario"
            ]
            == PRIMARY_SCENARIO
        ][
            [
                "horizon_weeks",
                "MAE",
                "RMSE",
                "MASE",
            ]
        ]
        .rename(
            columns={
                "MAE":
                    "primary_MAE",

                "RMSE":
                    "primary_RMSE",

                "MASE":
                    "primary_MASE",
            }
        )
    )

    comparison = metrics.merge(
        primary_reference,
        on="horizon_weeks",
        how="left",
        validate="many_to_one",
    )

    comparison[
        "delta_MAE_vs_primary"
    ] = (
        comparison["MAE"]
        -
        comparison["primary_MAE"]
    )

    comparison[
        "delta_RMSE_vs_primary"
    ] = (
        comparison["RMSE"]
        -
        comparison["primary_RMSE"]
    )

    comparison[
        "delta_MASE_vs_primary"
    ] = (
        comparison["MASE"]
        -
        comparison["primary_MASE"]
    )

    comparison[
        "percent_change_MAE_vs_primary"
    ] = (
        comparison[
            "delta_MAE_vs_primary"
        ]
        /
        comparison[
            "primary_MAE"
        ]
        * 100
    )

    print(
        "\n" + "=" * 82
    )

    print(
        "CHANGE RELATIVE TO PRIMARY ANALYSIS"
    )

    print(
        "=" * 82
    )

    # IMPORTANT:
    # Sort BEFORE selecting display columns.
    # This fixes the previous KeyError.
    comparison_display = (
        comparison
        .sort_values(
            [
                "horizon_weeks",
                "covid_censored_value",
            ]
        )[
            [
                "scenario",
                "covid_censored_value",
                "horizon_weeks",
                "delta_MAE_vs_primary",
                "percent_change_MAE_vs_primary",
                "delta_RMSE_vs_primary",
                "delta_MASE_vs_primary",
            ]
        ]
    )

    print(
        comparison_display
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    SENSITIVITY_METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        SENSITIVITY_METRICS_FILE,
        index=False,
    )

    predictions.to_csv(
        SENSITIVITY_PREDICTIONS_FILE,
        index=False,
    )

    primary_reproduction.to_csv(
        PRIMARY_REPRODUCTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 82
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 82
    )

    print(
        "\nSensitivity metrics:\n"
        f"{SENSITIVITY_METRICS_FILE}"
    )

    print(
        "\nSensitivity predictions:\n"
        f"{SENSITIVITY_PREDICTIONS_FILE}"
    )

    print(
        "\nPrimary reproduction QC:\n"
        f"{PRIMARY_REPRODUCTION_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The frozen primary models "
        "were NOT re-selected."
    )

    print(
        "These results are sensitivity / "
        "robustness analyses only."
    )

    print(
        "Do NOT replace the primary model "
        "based on these sensitivity results."
    )


if __name__ == "__main__":
    main()