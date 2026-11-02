"""
28_multipathogen_model_interpretation.py

Interpretation of the final frozen multi-pathogen ARI
forecasting models.

PURPOSE
-------
1. Refit each frozen model using exactly the same
   development data used for the final Step 26 models.

2. Verify that the refitted models reproduce the Step 26
   held-out predictions.

3. Interpret the frozen 1-week Negative Binomial model
   using standardized coefficients and incidence-rate ratios.

4. Interpret the frozen XGBoost models using:
      - gain-based feature importance
      - SHAP contributions calculated directly by XGBoost

IMPORTANT
---------
This script does NOT select, tune, or replace models.

The frozen model specifications from Step 25 remain unchanged.

Interpretation is descriptive and must not be interpreted
as evidence of causality.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
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

XGB_VALIDATION_METRICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_validation_metrics.csv"
)

STEP26_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_final_test_predictions.csv"
)

NB_INTERPRETATION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_nb_coefficient_interpretation.csv"
)

XGB_GAIN_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_gain_importance.csv"
)

XGB_SHAP_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_shap_importance.csv"
)

XGB_SHAP_VALUES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_shap_values.csv"
)

REFIT_QC_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_interpretation_refit_qc.csv"
)


# ============================================================
# SETTINGS
# ============================================================

REFIT_TOLERANCE = 1e-4


# ============================================================
# FEATURE SETS
# These must match Steps 23-27.
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
# NEGATIVE BINOMIAL HELPERS
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
    evaluation,
    features,
):

    x_dev = (
        development[features]
        .astype(float)
        .copy()
    )

    x_eval = (
        evaluation[features]
        .astype(float)
        .copy()
    )

    y_dev = (
        development["target"]
        .astype(float)
        .to_numpy()
    )

    means = x_dev.mean()

    standard_deviations = (
        x_dev
        .std(ddof=0)
        .replace(
            0,
            1.0,
        )
    )

    x_dev_standardized = (
        x_dev - means
    ) / standard_deviations

    x_eval_standardized = (
        x_eval - means
    ) / standard_deviations

    x_dev_standardized = (
        sm.add_constant(
            x_dev_standardized,
            has_constant="add",
        )
    )

    x_eval_standardized = (
        sm.add_constant(
            x_eval_standardized,
            has_constant="add",
        )
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
            x_dev_standardized,
            family=family,
        )

        result = model.fit(
            maxiter=200,
            disp=0,
        )

    predictions = np.asarray(
        result.predict(
            x_eval_standardized
        ),
        dtype=float,
    )

    predictions = np.clip(
        predictions,
        0,
        None,
    )

    return (
        predictions,
        result,
        means,
        standard_deviations,
        alpha,
    )


# ============================================================
# XGBOOST CONFIGURATION RETRIEVAL
# ============================================================

def get_xgboost_configuration(
    validation_metrics,
    horizon,
    feature_set,
    configuration,
):

    if (
        "config_id"
        in validation_metrics.columns
    ):
        config_column = "config_id"

    elif (
        "configuration"
        in validation_metrics.columns
    ):
        config_column = "configuration"

    else:
        raise ValueError(
            "Could not find an XGBoost "
            "configuration column."
        )

    match = validation_metrics[
        (
            validation_metrics[
                "horizon_weeks"
            ]
            == horizon
        )
        &
        (
            validation_metrics[
                "feature_set"
            ]
            == feature_set
        )
        &
        (
            validation_metrics[
                config_column
            ].astype(str)
            == str(configuration)
        )
    ].copy()

    if match.empty:

        raise RuntimeError(
            "Could not locate XGBoost "
            "validation configuration for:\n"
            f"horizon={horizon}\n"
            f"feature_set={feature_set}\n"
            f"configuration={configuration}"
        )

    row = match.iloc[0]

    required_parameters = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    ]

    missing = [
        parameter
        for parameter
        in required_parameters
        if parameter
        not in match.columns
    ]

    if missing:
        raise ValueError(
            "Missing XGBoost parameter "
            f"columns: {missing}"
        )

    return {

        "n_estimators":
            int(
                row[
                    "n_estimators"
                ]
            ),

        "max_depth":
            int(
                row[
                    "max_depth"
                ]
            ),

        "learning_rate":
            float(
                row[
                    "learning_rate"
                ]
            ),

        "min_child_weight":
            float(
                row[
                    "min_child_weight"
                ]
            ),

        "subsample":
            float(
                row[
                    "subsample"
                ]
            ),

        "colsample_bytree":
            float(
                row[
                    "colsample_bytree"
                ]
            ),

        "reg_lambda":
            float(
                row[
                    "reg_lambda"
                ]
            ),
    }


# ============================================================
# XGBOOST FIT
# ============================================================

def fit_xgboost(
    development,
    evaluation,
    features,
    parameters,
):

    x_dev = (
        development[features]
        .astype(float)
    )

    y_dev = (
        development["target"]
        .astype(float)
        .to_numpy()
    )

    x_eval = (
        evaluation[features]
        .astype(float)
    )

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=(
            parameters[
                "n_estimators"
            ]
        ),

        max_depth=(
            parameters[
                "max_depth"
            ]
        ),

        learning_rate=(
            parameters[
                "learning_rate"
            ]
        ),

        min_child_weight=(
            parameters[
                "min_child_weight"
            ]
        ),

        subsample=(
            parameters[
                "subsample"
            ]
        ),

        colsample_bytree=(
            parameters[
                "colsample_bytree"
            ]
        ),

        reg_lambda=(
            parameters[
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

    predictions = np.asarray(
        model.predict(
            x_eval
        ),
        dtype=float,
    )

    predictions = np.clip(
        predictions,
        0,
        None,
    )

    return (
        predictions,
        model,
    )


# ============================================================
# NEGATIVE BINOMIAL INTERPRETATION
# ============================================================

def interpret_negative_binomial(
    result,
    features,
    horizon,
    feature_set,
    alpha,
):

    confidence_intervals = (
        result.conf_int()
    )

    rows = []

    parameter_names = (
        list(
            result.params.index
        )
    )

    for parameter in parameter_names:

        beta = float(
            result.params[
                parameter
            ]
        )

        p_value = float(
            result.pvalues[
                parameter
            ]
        )

        lower_beta = float(
            confidence_intervals.loc[
                parameter,
                0,
            ]
        )

        upper_beta = float(
            confidence_intervals.loc[
                parameter,
                1,
            ]
        )

        irr = float(
            np.exp(beta)
        )

        irr_lower = float(
            np.exp(
                lower_beta
            )
        )

        irr_upper = float(
            np.exp(
                upper_beta
            )
        )

        percent_change = float(
            (
                irr - 1
            )
            * 100
        )

        rows.append(
            {
                "horizon_weeks":
                    horizon,

                "feature_set":
                    feature_set,

                "parameter":
                    parameter,

                "beta_standardized":
                    beta,

                "IRR_per_1SD":
                    irr,

                "IRR_95CI_lower":
                    irr_lower,

                "IRR_95CI_upper":
                    irr_upper,

                "percent_change_per_1SD":
                    percent_change,

                "p_value":
                    p_value,

                "dispersion_alpha":
                    alpha,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# XGBOOST GAIN IMPORTANCE
# ============================================================

def calculate_gain_importance(
    model,
    features,
    horizon,
    feature_set,
    configuration,
):

    booster = (
        model.get_booster()
    )

    raw_gain = (
        booster.get_score(
            importance_type="gain"
        )
    )

    rows = []

    gains = []

    for index, feature in enumerate(
        features
    ):

        value = raw_gain.get(
            feature,
            raw_gain.get(
                f"f{index}",
                0.0,
            ),
        )

        value = float(
            value
        )

        gains.append(
            value
        )

    total_gain = sum(
        gains
    )

    for feature, gain_value in zip(
        features,
        gains,
    ):

        if total_gain > 0:

            normalized_gain = (
                gain_value
                / total_gain
            )

        else:

            normalized_gain = 0.0

        rows.append(
            {
                "horizon_weeks":
                    horizon,

                "feature_set":
                    feature_set,

                "configuration":
                    configuration,

                "feature":
                    feature,

                "gain":
                    gain_value,

                "normalized_gain":
                    normalized_gain,
            }
        )

    importance = pd.DataFrame(
        rows
    )

    importance = (
        importance
        .sort_values(
            "normalized_gain",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance[
        "gain_rank"
    ] = (
        np.arange(
            1,
            len(importance) + 1,
        )
    )

    return importance


# ============================================================
# XGBOOST SHAP CONTRIBUTIONS
# ============================================================

def calculate_shap_values(
    model,
    dataset,
    features,
    horizon,
    feature_set,
    configuration,
    dataset_part,
):

    x_values = (
        dataset[features]
        .astype(float)
        .copy()
    )

    booster = (
        model.get_booster()
    )

    matrix = xgb.DMatrix(
        x_values,
        feature_names=features,
    )

    contributions = (
        booster.predict(
            matrix,
            pred_contribs=True,
        )
    )

    # Final contribution column is the model bias.
    shap_values = (
        contributions[
            :,
            :-1
        ]
    )

    bias = (
        contributions[
            :,
            -1
        ]
    )

    # --------------------------------------------------------
    # GLOBAL SHAP IMPORTANCE
    # --------------------------------------------------------

    global_rows = []

    for index, feature in enumerate(
        features
    ):

        values = (
            shap_values[
                :,
                index
            ]
        )

        global_rows.append(
            {
                "horizon_weeks":
                    horizon,

                "feature_set":
                    feature_set,

                "configuration":
                    configuration,

                "dataset_part":
                    dataset_part,

                "feature":
                    feature,

                "mean_abs_shap":
                    float(
                        np.mean(
                            np.abs(
                                values
                            )
                        )
                    ),

                "mean_shap":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "median_abs_shap":
                    float(
                        np.median(
                            np.abs(
                                values
                            )
                        )
                    ),
            }
        )

    global_importance = (
        pd.DataFrame(
            global_rows
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    global_importance[
        "shap_rank"
    ] = np.arange(
        1,
        len(
            global_importance
        ) + 1,
    )

    # --------------------------------------------------------
    # ROW-LEVEL SHAP VALUES
    # --------------------------------------------------------

    key_columns = [
        column
        for column in [
            "year",
            "epi_week",
            "target_year",
            "target_epi_week",
            "horizon_weeks",
            "target",
        ]
        if column in dataset.columns
    ]

    detailed = (
        dataset[
            key_columns
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    detailed[
        "dataset_part"
    ] = dataset_part

    detailed[
        "feature_set"
    ] = feature_set

    detailed[
        "configuration"
    ] = configuration

    detailed[
        "model_bias"
    ] = bias

    for index, feature in enumerate(
        features
    ):

        detailed[
            f"value__{feature}"
        ] = (
            x_values[
                feature
            ]
            .to_numpy()
        )

        detailed[
            f"shap__{feature}"
        ] = (
            shap_values[
                :,
                index
            ]
        )

    return (
        global_importance,
        detailed,
    )


# ============================================================
# STEP 26 REPRODUCTION CHECK
# ============================================================

def compare_predictions_with_step26(
    current_predictions,
    step26_predictions,
    horizon,
):

    previous = (
        step26_predictions[
            step26_predictions[
                "horizon_weeks"
            ]
            == horizon
        ]
        .copy()
    )

    current = (
        current_predictions
        .copy()
    )

    merge_keys = [
        "year",
        "epi_week",
        "target_year",
        "target_epi_week",
        "horizon_weeks",
    ]

    merge_keys = [
        column
        for column in merge_keys
        if (
            column
            in previous.columns
            and
            column
            in current.columns
        )
    ]

    if not merge_keys:

        raise RuntimeError(
            "No common keys available "
            "for Step 26 reproduction QC."
        )

    previous = previous.rename(
        columns={
            "prediction":
                "step26_prediction"
        }
    )

    current = current.rename(
        columns={
            "prediction":
                "step28_prediction"
        }
    )

    comparison = current.merge(
        previous[
            merge_keys
            + [
                "step26_prediction"
            ]
        ],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(current):

        raise RuntimeError(
            f"Prediction QC row mismatch "
            f"for horizon {horizon}."
        )

    comparison[
        "absolute_prediction_difference"
    ] = np.abs(
        comparison[
            "step28_prediction"
        ]
        -
        comparison[
            "step26_prediction"
        ]
    )

    maximum_difference = float(
        comparison[
            "absolute_prediction_difference"
        ]
        .max()
    )

    return (
        maximum_difference,
        len(comparison),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print("FINAL FROZEN MODEL INTERPRETATION")
    print("=" * 82)

    required_files = [
        DATA_FILE,
        SELECTION_FILE,
        XGB_VALIDATION_METRICS_FILE,
        STEP26_PREDICTIONS_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    data = pd.read_csv(
        DATA_FILE
    )

    selection = pd.read_csv(
        SELECTION_FILE
    )

    xgb_validation_metrics = pd.read_csv(
        XGB_VALIDATION_METRICS_FILE
    )

    step26_predictions = pd.read_csv(
        STEP26_PREDICTIONS_FILE
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
            [
                "horizon_weeks",
                "model_family",
                "feature_set",
                "configuration",
            ]
        ]
        .to_string(
            index=False
        )
    )

    nb_tables = []
    gain_tables = []
    shap_importance_tables = []
    shap_value_tables = []
    qc_rows = []

    # --------------------------------------------------------
    # LOOP THROUGH FROZEN MODELS
    # --------------------------------------------------------

    for _, selected in (
        selection.iterrows()
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
                f"Unknown feature set: "
                f"{feature_set}"
            )

        features = FEATURE_SETS[
            feature_set
        ]

        horizon_data = (
            data[
                data[
                    "horizon_weeks"
                ]
                == horizon
            ]
            .copy()
        )

        development = (
            horizon_data[
                horizon_data[
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
            horizon_data[
                horizon_data[
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

        required_columns = (
            features
            + ["target"]
        )

        if (
            development[
                required_columns
            ]
            .isna()
            .any()
            .any()
        ):

            raise RuntimeError(
                f"Missing development values "
                f"at horizon {horizon}."
            )

        if (
            test[
                required_columns
            ]
            .isna()
            .any()
            .any()
        ):

            raise RuntimeError(
                f"Missing test values "
                f"at horizon {horizon}."
            )

        print(
            "\n" + "=" * 82
        )

        print(
            f"HORIZON {horizon} WEEK(S)"
        )

        print(
            "=" * 82
        )

        print(
            f"Model family: "
            f"{model_family}"
        )

        print(
            f"Feature set: "
            f"{feature_set}"
        )

        print(
            f"Configuration: "
            f"{configuration}"
        )

        # ====================================================
        # NEGATIVE BINOMIAL
        # ====================================================

        if (
            model_family
            == "negative_binomial"
        ):

            (
                prediction,
                result,
                feature_means,
                feature_sds,
                alpha,
            ) = (
                fit_negative_binomial(
                    development=development,
                    evaluation=test,
                    features=features,
                )
            )

            interpretation = (
                interpret_negative_binomial(
                    result=result,
                    features=features,
                    horizon=horizon,
                    feature_set=feature_set,
                    alpha=alpha,
                )
            )

            nb_tables.append(
                interpretation
            )

            print(
                "\nNEGATIVE BINOMIAL "
                "COEFFICIENT INTERPRETATION"
            )

            print(
                interpretation[
                    interpretation[
                        "parameter"
                    ]
                    != "const"
                ][
                    [
                        "parameter",
                        "beta_standardized",
                        "IRR_per_1SD",
                        "IRR_95CI_lower",
                        "IRR_95CI_upper",
                        "percent_change_per_1SD",
                        "p_value",
                    ]
                ]
                .round(4)
                .to_string(
                    index=False
                )
            )

        # ====================================================
        # XGBOOST
        # ====================================================

        elif (
            model_family
            == "xgboost"
        ):

            parameters = (
                get_xgboost_configuration(
                    validation_metrics=(
                        xgb_validation_metrics
                    ),
                    horizon=horizon,
                    feature_set=feature_set,
                    configuration=configuration,
                )
            )

            print(
                "\nFrozen XGBoost parameters:"
            )

            for key, value in (
                parameters.items()
            ):

                print(
                    f"  {key}: {value}"
                )

            (
                prediction,
                model,
            ) = fit_xgboost(
                development=development,
                evaluation=test,
                features=features,
                parameters=parameters,
            )

            # --------------------------------------------
            # GAIN IMPORTANCE
            # --------------------------------------------

            gain_importance = (
                calculate_gain_importance(
                    model=model,
                    features=features,
                    horizon=horizon,
                    feature_set=feature_set,
                    configuration=configuration,
                )
            )

            gain_tables.append(
                gain_importance
            )

            print(
                "\nXGBOOST GAIN IMPORTANCE"
            )

            print(
                gain_importance[
                    [
                        "gain_rank",
                        "feature",
                        "gain",
                        "normalized_gain",
                    ]
                ]
                .round(4)
                .to_string(
                    index=False
                )
            )

            # --------------------------------------------
            # SHAP: DEVELOPMENT
            # --------------------------------------------

            (
                development_shap_importance,
                development_shap_values,
            ) = calculate_shap_values(
                model=model,
                dataset=development,
                features=features,
                horizon=horizon,
                feature_set=feature_set,
                configuration=configuration,
                dataset_part="development",
            )

            shap_importance_tables.append(
                development_shap_importance
            )

            shap_value_tables.append(
                development_shap_values
            )

            # --------------------------------------------
            # SHAP: HELD-OUT 2026
            # Descriptive only.
            # --------------------------------------------

            (
                test_shap_importance,
                test_shap_values,
            ) = calculate_shap_values(
                model=model,
                dataset=test,
                features=features,
                horizon=horizon,
                feature_set=feature_set,
                configuration=configuration,
                dataset_part="test_2026",
            )

            shap_importance_tables.append(
                test_shap_importance
            )

            shap_value_tables.append(
                test_shap_values
            )

            print(
                "\nXGBOOST SHAP IMPORTANCE "
                "— DEVELOPMENT DATA"
            )

            print(
                development_shap_importance[
                    [
                        "shap_rank",
                        "feature",
                        "mean_abs_shap",
                        "mean_shap",
                    ]
                ]
                .round(4)
                .to_string(
                    index=False
                )
            )

            print(
                "\nXGBOOST SHAP IMPORTANCE "
                "— 2026 TEST DATA"
            )

            print(
                test_shap_importance[
                    [
                        "shap_rank",
                        "feature",
                        "mean_abs_shap",
                        "mean_shap",
                    ]
                ]
                .round(4)
                .to_string(
                    index=False
                )
            )

        else:

            raise ValueError(
                f"Unknown model family: "
                f"{model_family}"
            )

        # ----------------------------------------------------
        # STEP 26 PREDICTION REPRODUCTION QC
        # ----------------------------------------------------

        current_predictions = (
            test[
                [
                    "year",
                    "epi_week",
                    "target_year",
                    "target_epi_week",
                    "horizon_weeks",
                ]
            ]
            .copy()
        )

        current_predictions[
            "prediction"
        ] = prediction

        (
            max_difference,
            matched_rows,
        ) = (
            compare_predictions_with_step26(
                current_predictions=(
                    current_predictions
                ),
                step26_predictions=(
                    step26_predictions
                ),
                horizon=horizon,
            )
        )

        qc_rows.append(
            {
                "horizon_weeks":
                    horizon,

                "model_family":
                    model_family,

                "feature_set":
                    feature_set,

                "configuration":
                    configuration,

                "matched_test_rows":
                    matched_rows,

                "max_abs_prediction_difference":
                    max_difference,

                "passed":
                    (
                        max_difference
                        <= REFIT_TOLERANCE
                    ),
            }
        )

        print(
            "\nSTEP 26 REPRODUCTION QC"
        )

        print(
            "Maximum absolute prediction "
            f"difference: {max_difference:.8f}"
        )

        if (
            max_difference
            > REFIT_TOLERANCE
        ):

            raise RuntimeError(
                "\nInterpretation refit does "
                "not reproduce Step 26.\n"
                f"Horizon: {horizon}\n"
                f"Maximum difference: "
                f"{max_difference}\n"
                "Do not interpret results "
                "until this is corrected."
            )

        print(
            "REFIT REPRODUCTION PASSED."
        )

    # ========================================================
    # COMBINE OUTPUTS
    # ========================================================

    qc = pd.DataFrame(
        qc_rows
    )

    if nb_tables:

        nb_interpretation = (
            pd.concat(
                nb_tables,
                ignore_index=True,
            )
        )

    else:

        nb_interpretation = (
            pd.DataFrame()
        )

    if gain_tables:

        xgb_gain = (
            pd.concat(
                gain_tables,
                ignore_index=True,
            )
        )

    else:

        xgb_gain = (
            pd.DataFrame()
        )

    if shap_importance_tables:

        shap_importance = (
            pd.concat(
                shap_importance_tables,
                ignore_index=True,
            )
        )

    else:

        shap_importance = (
            pd.DataFrame()
        )

    if shap_value_tables:

        shap_values = (
            pd.concat(
                shap_value_tables,
                ignore_index=True,
            )
        )

    else:

        shap_values = (
            pd.DataFrame()
        )

    # ========================================================
    # FINAL QC SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 82
    )

    print(
        "INTERPRETATION REFIT QC SUMMARY"
    )

    print(
        "=" * 82
    )

    print(
        qc.round(8)
        .to_string(
            index=False
        )
    )

    if not qc["passed"].all():

        raise RuntimeError(
            "At least one interpretation "
            "model failed Step 26 reproduction."
        )

    print(
        "\nALL FROZEN MODEL REFITS "
        "REPRODUCED STEP 26."
    )

    # ========================================================
    # SAVE
    # ========================================================

    NB_INTERPRETATION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nb_interpretation.to_csv(
        NB_INTERPRETATION_FILE,
        index=False,
    )

    xgb_gain.to_csv(
        XGB_GAIN_FILE,
        index=False,
    )

    shap_importance.to_csv(
        XGB_SHAP_IMPORTANCE_FILE,
        index=False,
    )

    shap_values.to_csv(
        XGB_SHAP_VALUES_FILE,
        index=False,
    )

    qc.to_csv(
        REFIT_QC_FILE,
        index=False,
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

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
        "\nNegative Binomial interpretation:\n"
        f"{NB_INTERPRETATION_FILE}"
    )

    print(
        "\nXGBoost gain importance:\n"
        f"{XGB_GAIN_FILE}"
    )

    print(
        "\nXGBoost SHAP importance:\n"
        f"{XGB_SHAP_IMPORTANCE_FILE}"
    )

    print(
        "\nXGBoost SHAP values:\n"
        f"{XGB_SHAP_VALUES_FILE}"
    )

    print(
        "\nRefit reproduction QC:\n"
        f"{REFIT_QC_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Interpretation was performed on "
        "the frozen models only."
    )

    print(
        "No model selection or tuning "
        "was performed."
    )

    print(
        "Feature importance and SHAP "
        "values describe prediction, "
        "not causal effects."
    )


if __name__ == "__main__":
    main()