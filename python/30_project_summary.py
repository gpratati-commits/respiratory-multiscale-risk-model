"""
30_project_summary.py

Generate a concise, reproducible summary of the final
multi-pathogen ARI forecasting project.

IMPORTANT
---------
This script does not fit, tune, select, or modify models.

It reads only frozen final outputs generated in earlier steps.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_FILE = (
    TABLE_DIR
    / "final_project_summary.txt"
)

MODEL_FILE = (
    TABLE_DIR
    / "final_table_1_model_specifications.csv"
)

TEST_FILE = (
    TABLE_DIR
    / "final_table_2_test_performance.csv"
)

SENSITIVITY_FILE = (
    TABLE_DIR
    / "final_table_3_sensitivity_analysis.csv"
)

NB_FILE = (
    TABLE_DIR
    / "final_table_4_negative_binomial_interpretation.csv"
)

SHAP_FILE = (
    TABLE_DIR
    / "final_table_5_xgboost_shap_importance.csv"
)

GAIN_FILE = (
    TABLE_DIR
    / "final_table_6_xgboost_gain_importance.csv"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def format_table(df, decimals=3):

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    result = df.copy()

    result[numeric_columns] = (
        result[numeric_columns]
        .round(decimals)
    )

    return result.to_string(
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    required = [
        MODEL_FILE,
        TEST_FILE,
        SENSITIVITY_FILE,
        NB_FILE,
        SHAP_FILE,
        GAIN_FILE,
    ]

    for path in required:
        require_file(path)

    models = pd.read_csv(
        MODEL_FILE
    )

    test = pd.read_csv(
        TEST_FILE
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_FILE
    )

    nb = pd.read_csv(
        NB_FILE
    )

    shap = pd.read_csv(
        SHAP_FILE
    )

    gain = pd.read_csv(
        GAIN_FILE
    )

    lines = []

    def add(text=""):
        lines.append(str(text))

    # ========================================================
    # HEADER
    # ========================================================

    add("=" * 80)
    add("FINAL PROJECT SUMMARY")
    add("MULTI-PATHOGEN ARI FORECASTING")
    add("=" * 80)

    add()
    add(
        "This summary contains only frozen final results. "
        "No model fitting or model selection is performed here."
    )

    # ========================================================
    # FINAL MODEL SPECIFICATIONS
    # ========================================================

    add()
    add("=" * 80)
    add("FINAL MODEL SPECIFICATIONS")
    add("=" * 80)

    add(
        format_table(
            models
        )
    )

    # ========================================================
    # HELD-OUT RESULTS
    # ========================================================

    add()
    add("=" * 80)
    add("FINAL 2026 HELD-OUT PERFORMANCE")
    add("=" * 80)

    add(
        format_table(
            test
        )
    )

    add()
    add("Key held-out interpretation:")

    for _, row in (
        test
        .sort_values(
            "horizon_weeks"
        )
        .iterrows()
    ):

        add(
            f"- {int(row['horizon_weeks'])}-week: "
            f"{row['model_family']} / "
            f"{row['feature_set']} | "
            f"MAE={row['MAE']:.3f}, "
            f"RMSE={row['RMSE']:.3f}, "
            f"MASE={row['MASE']:.3f}"
        )

    add()
    add(
        "Forecast accuracy decreased as forecast horizon increased."
    )

    add(
        "The 1-week model achieved MASE below 1, while the "
        "2- and 4-week models had MASE above 1."
    )

    # ========================================================
    # SENSITIVITY
    # ========================================================

    add()
    add("=" * 80)
    add("COVID-19 CENSORING SENSITIVITY ANALYSIS")
    add("=" * 80)

    add(
        format_table(
            sensitivity
        )
    )

    add()
    add(
        "Interpretation: alternative representations of "
        "source-reported COVID-19 positivity <1% did not "
        "materially alter the frozen-model results."
    )

    # ========================================================
    # NEGATIVE BINOMIAL INTERPRETATION
    # ========================================================

    add()
    add("=" * 80)
    add("NEGATIVE BINOMIAL MODEL INTERPRETATION")
    add("=" * 80)

    add(
        format_table(
            nb
        )
    )

    add()
    add(
        "IRRs describe associations within the forecasting model "
        "and must not be interpreted as causal effects."
    )

    # ========================================================
    # XGBOOST SHAP
    # ========================================================

    add()
    add("=" * 80)
    add("XGBOOST SHAP IMPORTANCE")
    add("=" * 80)

    add(
        format_table(
            shap
        )
    )

    # --------------------------------------------------------
    # 2-WEEK MODEL
    # --------------------------------------------------------

    shap_2 = (
        shap[
            shap[
                "horizon_weeks"
            ]
            == 2
        ]
        .sort_values(
            "shap_rank"
        )
        .copy()
    )

    if not shap_2.empty:

        add()
        add("2-week model interpretation:")

        for _, row in (
            shap_2.iterrows()
        ):

            add(
                f"- Rank {int(row['shap_rank'])}: "
                f"{row['feature']} "
                f"(mean |SHAP| = "
                f"{row['mean_abs_shap']:.3f})"
            )

    # --------------------------------------------------------
    # 4-WEEK MODEL
    # --------------------------------------------------------

    shap_4 = (
        shap[
            shap[
                "horizon_weeks"
            ]
            == 4
        ]
        .sort_values(
            "shap_rank"
        )
        .copy()
    )

    if not shap_4.empty:

        add()
        add("4-week model interpretation:")

        for _, row in (
            shap_4.iterrows()
        ):

            add(
                f"- Rank {int(row['shap_rank'])}: "
                f"{row['feature']} "
                f"(mean |SHAP| = "
                f"{row['mean_abs_shap']:.3f})"
            )

        covid_rows = (
            shap_4[
                shap_4[
                    "feature"
                ]
                .str.contains(
                    "covid",
                    case=False,
                    na=False,
                )
            ]
        )

        if (
            not covid_rows.empty
            and
            (
                covid_rows[
                    "mean_abs_shap"
                ]
                == 0
            )
            .all()
        ):

            add()
            add(
                "Important finding: although the ARI + COVID "
                "feature set was selected for the 4-week horizon, "
                "the COVID predictors had zero mean absolute SHAP "
                "importance in the frozen fitted model."
            )

            add(
                "The 4-week predictions were therefore driven "
                "primarily by seasonality and ARI-history predictors."
            )

    # ========================================================
    # XGBOOST GAIN
    # ========================================================

    add()
    add("=" * 80)
    add("XGBOOST GAIN IMPORTANCE")
    add("=" * 80)

    add(
        format_table(
            gain
        )
    )

    # ========================================================
    # CORE SCIENTIFIC CONCLUSIONS
    # ========================================================

    add()
    add("=" * 80)
    add("CORE SCIENTIFIC CONCLUSIONS")
    add("=" * 80)

    add(
        "1. Short-horizon ARI forecasting was most accurate."
    )

    add(
        "2. The 1-week Negative Binomial ARI-only model achieved "
        "the strongest final held-out performance."
    )

    add(
        "3. The 2-week selected model was an ARI-only XGBoost model."
    )

    add(
        "4. The 4-week validation-selected model used an "
        "ARI + COVID feature set."
    )

    add(
        "5. However, post-hoc SHAP interpretation indicated that "
        "the included COVID predictors contributed little or no "
        "importance to the final 4-week predictions."
    )

    add(
        "6. Longer forecast horizons showed reduced predictive skill."
    )

    add(
        "7. COVID censoring assumptions did not materially change "
        "the frozen-model results."
    )

    # ========================================================
    # CAUTIONS
    # ========================================================

    add()
    add("=" * 80)
    add("INTERPRETATION CAUTIONS")
    add("=" * 80)

    add(
        "- Validation-selected feature sets should not automatically "
        "be interpreted as evidence that every included predictor "
        "contributed materially."
    )

    add(
        "- SHAP and gain importance describe predictive model "
        "behaviour, not causal effects."
    )

    add(
        "- Held-out 2026 performance must not be used for further "
        "model tuning or model re-selection."
    )

    add(
        "- Forecast skill decreased at longer horizons, and MASE "
        "exceeded 1 for the 2- and 4-week horizons."
    )

    # ========================================================
    # WRITE FILE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 80)
    print("FINAL PROJECT SUMMARY CREATED")
    print("=" * 80)

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "No model was fitted, tuned, or re-selected."
    )


if __name__ == "__main__":
    main()