"""
29_final_figures_tables.py

Create final publication-ready figures and summary tables
for the multi-pathogen ARI forecasting project.

IMPORTANT
---------
This script DOES NOT fit, tune, select, or modify any model.

It only reads the frozen results generated in Steps 20-28
and converts them into final figures and manuscript tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "final"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

WEEKLY_FILE = (
    DATA_DIR
    / "multipathogen_weekly_dataset.csv"
)

FINAL_SELECTION_FILE = (
    TABLE_DIR
    / "multipathogen_final_validation_selection.csv"
)

FINAL_TEST_METRICS_FILE = (
    TABLE_DIR
    / "multipathogen_final_test_metrics.csv"
)

FINAL_TEST_PREDICTIONS_FILE = (
    TABLE_DIR
    / "multipathogen_final_test_predictions.csv"
)

SENSITIVITY_FILE = (
    TABLE_DIR
    / "multipathogen_covid_censoring_sensitivity.csv"
)

NB_INTERPRETATION_FILE = (
    TABLE_DIR
    / "multipathogen_nb_coefficient_interpretation.csv"
)

XGB_GAIN_FILE = (
    TABLE_DIR
    / "multipathogen_xgboost_gain_importance.csv"
)

XGB_SHAP_FILE = (
    TABLE_DIR
    / "multipathogen_xgboost_shap_importance.csv"
)

REFIT_QC_FILE = (
    TABLE_DIR
    / "multipathogen_interpretation_refit_qc.csv"
)


# ------------------------------------------------------------
# FINAL TABLE OUTPUTS
# ------------------------------------------------------------

TABLE_1_FILE = (
    TABLE_DIR
    / "final_table_1_model_specifications.csv"
)

TABLE_2_FILE = (
    TABLE_DIR
    / "final_table_2_test_performance.csv"
)

TABLE_3_FILE = (
    TABLE_DIR
    / "final_table_3_sensitivity_analysis.csv"
)

TABLE_4_FILE = (
    TABLE_DIR
    / "final_table_4_negative_binomial_interpretation.csv"
)

TABLE_5_FILE = (
    TABLE_DIR
    / "final_table_5_xgboost_shap_importance.csv"
)

TABLE_6_FILE = (
    TABLE_DIR
    / "final_table_6_xgboost_gain_importance.csv"
)

FIGURE_MANIFEST_FILE = (
    TABLE_DIR
    / "final_figure_manifest.csv"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            "Required file not found:\n"
            f"{path}"
        )


def save_figure(path):

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def clean_feature_name(name):

    replacements = {

        "ari_current":
            "Current ARI attendance",

        "ari_recent_4week_mean":
            "Recent 4-week ARI mean",

        "ari_recent_2week_mean":
            "Recent 2-week ARI mean",

        "ari_lag1":
            "ARI lag 1 week",

        "ari_lag2":
            "ARI lag 2 weeks",

        "ari_lag4":
            "ARI lag 4 weeks",

        "influenza_lag1":
            "Influenza lag 1 week",

        "influenza_lag2":
            "Influenza lag 2 weeks",

        "influenza_lag4":
            "Influenza lag 4 weeks",

        "influenza_recent_4week_mean":
            "Recent 4-week influenza mean",

        "influenza_recent_2week_mean":
            "Recent 2-week influenza mean",

        "covid_lag1":
            "COVID-19 lag 1 week",

        "covid_lag2":
            "COVID-19 lag 2 weeks",

        "covid_lag4":
            "COVID-19 lag 4 weeks",

        "covid_recent_4week_mean":
            "Recent 4-week COVID-19 mean",

        "covid_recent_2week_mean":
            "Recent 2-week COVID-19 mean",

        "season_sin":
            "Seasonality (sine)",

        "season_cos":
            "Seasonality (cosine)",

        "const":
            "Intercept",
    }

    return replacements.get(
        str(name),
        str(name).replace(
            "_",
            " ",
        ),
    )


def get_prediction_column(df):

    candidates = [
        "prediction",
        "predicted",
        "y_pred",
        "forecast",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify prediction column "
        "in final test predictions."
    )


def get_target_column(df):

    candidates = [
        "target",
        "observed",
        "y_true",
        "actual",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify observed target column "
        "in final test predictions."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("STEP 29 — FINAL FIGURES AND TABLES")
    print("=" * 90)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------------

    input_files = [

        WEEKLY_FILE,

        FINAL_SELECTION_FILE,

        FINAL_TEST_METRICS_FILE,

        FINAL_TEST_PREDICTIONS_FILE,

        SENSITIVITY_FILE,

        NB_INTERPRETATION_FILE,

        XGB_GAIN_FILE,

        XGB_SHAP_FILE,

        REFIT_QC_FILE,
    ]

    for path in input_files:
        require_file(path)

    print(
        "\nAll required Step 29 input files were found."
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    weekly = pd.read_csv(
        WEEKLY_FILE
    )

    final_selection = pd.read_csv(
        FINAL_SELECTION_FILE
    )

    final_metrics = pd.read_csv(
        FINAL_TEST_METRICS_FILE
    )

    predictions = pd.read_csv(
        FINAL_TEST_PREDICTIONS_FILE
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_FILE
    )

    nb_interpretation = pd.read_csv(
        NB_INTERPRETATION_FILE
    )

    xgb_gain = pd.read_csv(
        XGB_GAIN_FILE
    )

    xgb_shap = pd.read_csv(
        XGB_SHAP_FILE
    )

    refit_qc = pd.read_csv(
        REFIT_QC_FILE
    )

    # ========================================================
    # QC
    # ========================================================

    if (
        "passed"
        in refit_qc.columns
    ):

        passed = (
            refit_qc[
                "passed"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
        )

        if not passed.all():

            raise RuntimeError(
                "Step 28 reproduction QC did not "
                "pass for every frozen model."
            )

    print(
        "Step 28 reproduction QC is valid."
    )

    # ========================================================
    # TABLE 1
    # FINAL FROZEN MODEL SPECIFICATIONS
    # ========================================================

    table_1_columns = [

        column
        for column in [

            "horizon_weeks",
            "model_family",
            "feature_set",
            "configuration",

        ]
        if column
        in final_selection.columns
    ]

    table_1 = (
        final_selection[
            table_1_columns
        ]
        .sort_values(
            "horizon_weeks"
        )
        .reset_index(
            drop=True
        )
    )

    table_1.to_csv(
        TABLE_1_FILE,
        index=False,
    )

    # ========================================================
    # TABLE 2
    # FINAL 2026 TEST PERFORMANCE
    # ========================================================

    table_2_columns = [

        column
        for column in [

            "horizon_weeks",
            "model_family",
            "feature_set",
            "configuration",
            "n_test",
            "MAE",
            "RMSE",
            "MASE",

        ]
        if column
        in final_metrics.columns
    ]

    table_2 = (
        final_metrics[
            table_2_columns
        ]
        .sort_values(
            "horizon_weeks"
        )
        .reset_index(
            drop=True
        )
    )

    table_2.to_csv(
        TABLE_2_FILE,
        index=False,
    )

    # ========================================================
    # TABLE 3
    # SENSITIVITY ANALYSIS
    # ========================================================

    sensitivity_columns = [

        column
        for column in [

            "scenario",
            "covid_censored_value",
            "horizon_weeks",
            "model_family",
            "feature_set",
            "MAE",
            "RMSE",
            "MASE",
            "delta_MAE_vs_primary",
            "percent_change_MAE_vs_primary",
            "delta_RMSE_vs_primary",
            "delta_MASE_vs_primary",

        ]
        if column
        in sensitivity.columns
    ]

    table_3 = (
        sensitivity[
            sensitivity_columns
        ]
        .sort_values(
            [
                "horizon_weeks",
                "covid_censored_value",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    table_3.to_csv(
        TABLE_3_FILE,
        index=False,
    )

    # ========================================================
    # TABLE 4
    # NEGATIVE BINOMIAL INTERPRETATION
    # ========================================================

    nb_columns = [

        column
        for column in [

            "horizon_weeks",
            "feature_set",
            "parameter",
            "beta_standardized",
            "IRR_per_1SD",
            "IRR_95CI_lower",
            "IRR_95CI_upper",
            "percent_change_per_1SD",
            "p_value",
            "dispersion_alpha",

        ]
        if column
        in nb_interpretation.columns
    ]

    table_4 = (
        nb_interpretation[
            nb_columns
        ]
        .copy()
    )

    table_4.to_csv(
        TABLE_4_FILE,
        index=False,
    )

    # ========================================================
    # TABLE 5
    # SHAP IMPORTANCE
    # ========================================================

    if (
        "dataset_part"
        in xgb_shap.columns
    ):

        primary_shap = (
            xgb_shap[
                xgb_shap[
                    "dataset_part"
                ]
                == "development"
            ]
            .copy()
        )

    else:

        primary_shap = (
            xgb_shap.copy()
        )

    shap_columns = [

        column
        for column in [

            "horizon_weeks",
            "feature_set",
            "configuration",
            "feature",
            "mean_abs_shap",
            "mean_shap",
            "median_abs_shap",
            "shap_rank",

        ]
        if column
        in primary_shap.columns
    ]

    table_5 = (
        primary_shap[
            shap_columns
        ]
        .sort_values(
            [
                "horizon_weeks",
                "shap_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    table_5.to_csv(
        TABLE_5_FILE,
        index=False,
    )

    # ========================================================
    # TABLE 6
    # XGBOOST GAIN IMPORTANCE
    # ========================================================

    gain_columns = [

        column
        for column in [

            "horizon_weeks",
            "feature_set",
            "configuration",
            "feature",
            "gain",
            "normalized_gain",
            "gain_rank",

        ]
        if column
        in xgb_gain.columns
    ]

    table_6 = (
        xgb_gain[
            gain_columns
        ]
        .sort_values(
            [
                "horizon_weeks",
                "gain_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    table_6.to_csv(
        TABLE_6_FILE,
        index=False,
    )

    # ========================================================
    # FIGURE MANIFEST
    # ========================================================

    figure_manifest = []

    # ========================================================
    # FIGURE 1
    # COMPLETE WEEKLY ARI SERIES
    # ========================================================

    weekly = (
        weekly
        .sort_values(
            [
                "year",
                "epi_week",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        "ari_attendance"
        not in weekly.columns
    ):

        raise ValueError(
            "ari_attendance column is missing "
            "from weekly dataset."
        )

    weekly[
        "sequential_week"
    ] = np.arange(
        1,
        len(weekly) + 1,
    )

    figure_1 = (
        FIGURE_DIR
        / "figure_1_weekly_ari_surveillance.png"
    )

    plt.figure(
        figsize=(
            12,
            6,
        )
    )

    plt.plot(
        weekly[
            "sequential_week"
        ],
        weekly[
            "ari_attendance"
        ],
        linewidth=1.5,
    )

    year_starts = (
        weekly
        .groupby(
            "year",
            as_index=False,
        )
        .first()
    )

    for _, row in (
        year_starts.iterrows()
    ):

        plt.axvline(
            row[
                "sequential_week"
            ],
            linewidth=0.7,
            alpha=0.4,
        )

        plt.text(
            row[
                "sequential_week"
            ],
            plt.ylim()[1],
            str(
                int(
                    row[
                        "year"
                    ]
                )
            ),
            rotation=90,
            va="top",
            fontsize=8,
        )

    plt.xlabel(
        "Sequential epidemiological week"
    )

    plt.ylabel(
        "Average daily ARI polyclinic attendance"
    )

    plt.title(
        "Weekly Acute Respiratory Infection "
        "Polyclinic Attendance"
    )

    save_figure(
        figure_1
    )

    figure_manifest.append(
        {
            "figure":
                "Figure 1",

            "file":
                figure_1.name,

            "description":
                "Complete weekly ARI surveillance series.",
        }
    )

    # ========================================================
    # FIGURES 2A-C
    # OBSERVED VS PREDICTED 2026
    # ========================================================

    prediction_column = (
        get_prediction_column(
            predictions
        )
    )

    target_column = (
        get_target_column(
            predictions
        )
    )

    horizons = sorted(
        predictions[
            "horizon_weeks"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    for horizon in horizons:

        sub = (
            predictions[
                predictions[
                    "horizon_weeks"
                ]
                == horizon
            ]
            .copy()
        )

        if (
            "target_epi_week"
            in sub.columns
        ):

            sub = (
                sub
                .sort_values(
                    "target_epi_week"
                )
            )

            x = (
                sub[
                    "target_epi_week"
                ]
            )

            x_label = (
                "2026 target epidemiological week"
            )

        else:

            sub = (
                sub
                .reset_index(
                    drop=True
                )
            )

            x = (
                np.arange(
                    1,
                    len(sub) + 1,
                )
            )

            x_label = (
                "Sequential held-out observation"
            )

        figure_path = (
            FIGURE_DIR
            / (
                f"figure_2_{horizon}week_"
                "observed_vs_predicted.png"
            )
        )

        plt.figure(
            figsize=(
                10,
                5.5,
            )
        )

        plt.plot(
            x,
            sub[
                target_column
            ],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="Observed",
        )

        plt.plot(
            x,
            sub[
                prediction_column
            ],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="Predicted",
        )

        plt.xlabel(
            x_label
        )

        plt.ylabel(
            "Average daily ARI attendance"
        )

        plt.title(
            f"{horizon}-Week-Ahead "
            "ARI Forecasts in the "
            "2026 Held-Out Test Period"
        )

        plt.legend()

        save_figure(
            figure_path
        )

        figure_manifest.append(
            {
                "figure":
                    f"Figure 2 — {horizon}-week",

                "file":
                    figure_path.name,

                "description":
                    (
                        "Observed versus predicted "
                        f"{horizon}-week-ahead ARI."
                    ),
            }
        )

    # ========================================================
    # FIGURE 3A
    # MAE BY HORIZON
    # ========================================================

    metric_plot_data = (
        final_metrics
        .sort_values(
            "horizon_weeks"
        )
        .copy()
    )

    figure_3a = (
        FIGURE_DIR
        / "figure_3a_mae_by_horizon.png"
    )

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.plot(
        metric_plot_data[
            "horizon_weeks"
        ],
        metric_plot_data[
            "MAE"
        ],
        marker="o",
        linewidth=1.5,
    )

    plt.xticks(
        metric_plot_data[
            "horizon_weeks"
        ]
    )

    plt.xlabel(
        "Forecast horizon (weeks)"
    )

    plt.ylabel(
        "Mean absolute error"
    )

    plt.title(
        "Held-Out Forecast MAE by Horizon"
    )

    save_figure(
        figure_3a
    )

    figure_manifest.append(
        {
            "figure":
                "Figure 3A",

            "file":
                figure_3a.name,

            "description":
                "Held-out MAE by forecast horizon.",
        }
    )

    # ========================================================
    # FIGURE 3B
    # RMSE BY HORIZON
    # ========================================================

    figure_3b = (
        FIGURE_DIR
        / "figure_3b_rmse_by_horizon.png"
    )

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.plot(
        metric_plot_data[
            "horizon_weeks"
        ],
        metric_plot_data[
            "RMSE"
        ],
        marker="o",
        linewidth=1.5,
    )

    plt.xticks(
        metric_plot_data[
            "horizon_weeks"
        ]
    )

    plt.xlabel(
        "Forecast horizon (weeks)"
    )

    plt.ylabel(
        "Root mean squared error"
    )

    plt.title(
        "Held-Out Forecast RMSE by Horizon"
    )

    save_figure(
        figure_3b
    )

    figure_manifest.append(
        {
            "figure":
                "Figure 3B",

            "file":
                figure_3b.name,

            "description":
                "Held-out RMSE by forecast horizon.",
        }
    )

    # ========================================================
    # FIGURE 3C
    # MASE BY HORIZON
    # ========================================================

    figure_3c = (
        FIGURE_DIR
        / "figure_3c_mase_by_horizon.png"
    )

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.plot(
        metric_plot_data[
            "horizon_weeks"
        ],
        metric_plot_data[
            "MASE"
        ],
        marker="o",
        linewidth=1.5,
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xticks(
        metric_plot_data[
            "horizon_weeks"
        ]
    )

    plt.xlabel(
        "Forecast horizon (weeks)"
    )

    plt.ylabel(
        "Mean absolute scaled error"
    )

    plt.title(
        "Held-Out Forecast MASE by Horizon"
    )

    save_figure(
        figure_3c
    )

    figure_manifest.append(
        {
            "figure":
                "Figure 3C",

            "file":
                figure_3c.name,

            "description":
                (
                    "Held-out MASE by horizon; "
                    "reference line at MASE = 1."
                ),
        }
    )

    # ========================================================
    # FIGURE 4
    # NEGATIVE BINOMIAL IRR
    # ========================================================

    nb_plot = (
        nb_interpretation[
            nb_interpretation[
                "parameter"
            ]
            != "const"
        ]
        .copy()
    )

    if not nb_plot.empty:

        nb_plot[
            "display_feature"
        ] = (
            nb_plot[
                "parameter"
            ]
            .map(
                clean_feature_name
            )
        )

        nb_plot = (
            nb_plot
            .sort_values(
                "IRR_per_1SD"
            )
            .reset_index(
                drop=True
            )
        )

        y_position = (
            np.arange(
                len(nb_plot)
            )
        )

        lower_error = (
            nb_plot[
                "IRR_per_1SD"
            ]
            -
            nb_plot[
                "IRR_95CI_lower"
            ]
        )

        upper_error = (
            nb_plot[
                "IRR_95CI_upper"
            ]
            -
            nb_plot[
                "IRR_per_1SD"
            ]
        )

        figure_4 = (
            FIGURE_DIR
            / "figure_4_negative_binomial_irr.png"
        )

        plt.figure(
            figsize=(
                9,
                5.5,
            )
        )

        plt.errorbar(
            nb_plot[
                "IRR_per_1SD"
            ],
            y_position,
            xerr=np.vstack(
                [
                    lower_error,
                    upper_error,
                ]
            ),
            fmt="o",
            capsize=4,
        )

        plt.axvline(
            1.0,
            linestyle="--",
            linewidth=1,
        )

        plt.yticks(
            y_position,
            nb_plot[
                "display_feature"
            ],
        )

        plt.xlabel(
            "Incidence rate ratio per 1-SD increase"
        )

        plt.ylabel(
            "Predictor"
        )

        plt.title(
            "Negative Binomial Predictor Associations "
            "for the Frozen 1-Week Model"
        )

        save_figure(
            figure_4
        )

        figure_manifest.append(
            {
                "figure":
                    "Figure 4",

                "file":
                    figure_4.name,

                "description":
                    (
                        "Negative Binomial incidence "
                        "rate ratios with 95% confidence "
                        "intervals."
                    ),
            }
        )

    # ========================================================
    # FIGURE 5
    # XGBOOST SHAP IMPORTANCE
    # ========================================================

    shap_horizons = sorted(
        primary_shap[
            "horizon_weeks"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    for horizon in shap_horizons:

        shap_sub = (
            primary_shap[
                primary_shap[
                    "horizon_weeks"
                ]
                == horizon
            ]
            .copy()
        )

        if shap_sub.empty:
            continue

        shap_sub[
            "display_feature"
        ] = (
            shap_sub[
                "feature"
            ]
            .map(
                clean_feature_name
            )
        )

        shap_sub = (
            shap_sub
            .sort_values(
                "mean_abs_shap",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )

        figure_5 = (
            FIGURE_DIR
            / (
                f"figure_5_{horizon}week_"
                "xgboost_shap_importance.png"
            )
        )

        plt.figure(
            figsize=(
                9,
                5.5,
            )
        )

        plt.barh(
            shap_sub[
                "display_feature"
            ],
            shap_sub[
                "mean_abs_shap"
            ],
        )

        plt.xlabel(
            "Mean absolute SHAP value"
        )

        plt.ylabel(
            "Predictor"
        )

        plt.title(
            f"XGBoost Feature Importance "
            f"for the Frozen {horizon}-Week Model"
        )

        save_figure(
            figure_5
        )

        figure_manifest.append(
            {
                "figure":
                    (
                        f"Figure 5 — "
                        f"{horizon}-week"
                    ),

                "file":
                    figure_5.name,

                "description":
                    (
                        "Development-data mean absolute "
                        "SHAP importance for the frozen "
                        f"{horizon}-week XGBoost model."
                    ),
            }
        )

    # ========================================================
    # FIGURE 6
    # COVID CENSORING SENSITIVITY
    # ========================================================

    required_sensitivity_columns = {
        "covid_censored_value",
        "horizon_weeks",
        "MAE",
    }

    if required_sensitivity_columns.issubset(
        sensitivity.columns
    ):

        figure_6 = (
            FIGURE_DIR
            / (
                "figure_6_covid_censoring_"
                "sensitivity.png"
            )
        )

        plt.figure(
            figsize=(
                8,
                5.5,
            )
        )

        for horizon in sorted(
            sensitivity[
                "horizon_weeks"
            ]
            .dropna()
            .astype(int)
            .unique()
        ):

            subset = (
                sensitivity[
                    sensitivity[
                        "horizon_weeks"
                    ]
                    == horizon
                ]
                .sort_values(
                    "covid_censored_value"
                )
            )

            plt.plot(
                subset[
                    "covid_censored_value"
                ],
                subset[
                    "MAE"
                ],
                marker="o",
                linewidth=1.5,
                label=(
                    f"{horizon}-week"
                ),
            )

        plt.xlabel(
            "Value assigned to source-reported COVID-19 <1% (%)"
        )

        plt.ylabel(
            "Held-out MAE"
        )

        plt.title(
            "Sensitivity to Representation of "
            "Censored COVID-19 Positivity"
        )

        plt.legend(
            title="Forecast horizon"
        )

        save_figure(
            figure_6
        )

        figure_manifest.append(
            {
                "figure":
                    "Figure 6",

                "file":
                    figure_6.name,

                "description":
                    (
                        "Sensitivity of held-out MAE "
                        "to representation of COVID-19 "
                        "positivity reported as <1%."
                    ),
            }
        )

    # ========================================================
    # SAVE FIGURE MANIFEST
    # ========================================================

    figure_manifest = (
        pd.DataFrame(
            figure_manifest
        )
    )

    figure_manifest.to_csv(
        FIGURE_MANIFEST_FILE,
        index=False,
    )

    # ========================================================
    # FINAL OUTPUT SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL TABLES CREATED"
    )

    print(
        "=" * 90
    )

    for path in [

        TABLE_1_FILE,

        TABLE_2_FILE,

        TABLE_3_FILE,

        TABLE_4_FILE,

        TABLE_5_FILE,

        TABLE_6_FILE,

    ]:

        print(
            path
        )

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL FIGURES CREATED"
    )

    print(
        "=" * 90
    )

    for file_path in sorted(
        FIGURE_DIR.glob(
            "*.png"
        )
    ):

        print(
            file_path
        )

    print(
        "\nFigure manifest:"
    )

    print(
        FIGURE_MANIFEST_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No model was fitted, tuned, "
        "or selected in Step 29."
    )

    print(
        "All figures and tables were "
        "generated from previously frozen results."
    )


if __name__ == "__main__":
    main()