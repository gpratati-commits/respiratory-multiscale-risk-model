"""
25_compare_multipathogen_models.py

Compare validation-selected Negative Binomial and XGBoost
multi-pathogen forecasting models.

IMPORTANT
---------
This script uses validation results only.

The 2026 held-out test set is NOT accessed.

Outputs
-------
outputs/tables/multipathogen_validation_model_comparison.csv
outputs/tables/multipathogen_final_validation_selection.csv
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NB_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_nb_validation_metrics.csv"
)

XGB_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_xgboost_best_by_feature_set.csv"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_validation_model_comparison.csv"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "multipathogen_final_validation_selection.csv"
)


def main():

    print("=" * 78)
    print("MULTI-PATHOGEN VALIDATION MODEL COMPARISON")
    print("=" * 78)

    if not NB_FILE.exists():
        raise FileNotFoundError(
            f"Missing NB results:\n{NB_FILE}"
        )

    if not XGB_FILE.exists():
        raise FileNotFoundError(
            f"Missing XGBoost results:\n{XGB_FILE}"
        )

    nb = pd.read_csv(NB_FILE)
    xgb = pd.read_csv(XGB_FILE)

    # --------------------------------------------------------
    # PREPARE NEGATIVE BINOMIAL RESULTS
    # --------------------------------------------------------

    nb_compare = nb[
        [
            "horizon_weeks",
            "feature_set",
            "MAE",
            "RMSE",
            "MASE",
            "converged",
        ]
    ].copy()

    nb_compare["model_family"] = (
        "negative_binomial"
    )

    nb_compare["configuration"] = (
        "fixed_validation_specification"
    )

    # --------------------------------------------------------
    # PREPARE XGBOOST RESULTS
    # --------------------------------------------------------

    xgb_compare = xgb[
        [
            "horizon_weeks",
            "feature_set",
            "config_id",
            "MAE",
            "RMSE",
            "MASE",
        ]
    ].copy()

    xgb_compare["model_family"] = "xgboost"

    xgb_compare["configuration"] = (
        xgb_compare["config_id"]
        .astype(str)
    )

    xgb_compare["converged"] = True

    xgb_compare = xgb_compare.drop(
        columns=["config_id"]
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    comparison = pd.concat(
        [
            nb_compare,
            xgb_compare,
        ],
        ignore_index=True,
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

    print(
        "\n" + "=" * 78
    )
    print("FULL VALIDATION COMPARISON")
    print("=" * 78)

    print(
        comparison[
            [
                "horizon_weeks",
                "model_family",
                "feature_set",
                "configuration",
                "MAE",
                "RMSE",
                "MASE",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # BEST MODEL AT EACH HORIZON
    # --------------------------------------------------------

    eligible = comparison[
        comparison["converged"] == True
    ].copy()

    final_selection = (
        eligible
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
        "\n" + "=" * 78
    )
    print("BEST VALIDATION MODEL BY HORIZON")
    print("=" * 78)

    print(
        final_selection[
            [
                "horizon_weeks",
                "model_family",
                "feature_set",
                "configuration",
                "MAE",
                "RMSE",
                "MASE",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # PATHOGEN INCREMENTAL VALUE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 78
    )
    print("ARI-ONLY VS PATHOGEN-ENHANCED")
    print("=" * 78)

    for horizon in sorted(
        comparison["horizon_weeks"].unique()
    ):

        sub = comparison[
            comparison["horizon_weeks"]
            == horizon
        ].copy()

        best_ari = (
            sub[
                sub["feature_set"]
                == "ari_only"
            ]
            .sort_values("MAE")
            .iloc[0]
        )

        pathogen = sub[
            sub["feature_set"]
            != "ari_only"
        ]

        best_pathogen = (
            pathogen
            .sort_values("MAE")
            .iloc[0]
        )

        delta = (
            best_ari["MAE"]
            - best_pathogen["MAE"]
        )

        percent = (
            delta
            / best_ari["MAE"]
            * 100
        )

        print(
            f"\nHorizon {horizon} week(s)"
        )

        print(
            "Best ARI-only:"
        )

        print(
            f"  {best_ari['model_family']} / "
            f"{best_ari['feature_set']} / "
            f"MAE={best_ari['MAE']:.3f}"
        )

        print(
            "Best pathogen-enhanced:"
        )

        print(
            f"  {best_pathogen['model_family']} / "
            f"{best_pathogen['feature_set']} / "
            f"MAE={best_pathogen['MAE']:.3f}"
        )

        print(
            f"MAE improvement from pathogen data: "
            f"{delta:.3f}"
        )

        print(
            f"Relative improvement: "
            f"{percent:.2f}%"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    COMPARISON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    final_selection.to_csv(
        SELECTION_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 78
    )
    print("OUTPUT FILES")
    print("=" * 78)

    print(
        f"\nFull comparison:\n"
        f"{COMPARISON_FILE}"
    )

    print(
        f"\nFinal validation selection:\n"
        f"{SELECTION_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The 2026 held-out test set "
        "was NOT accessed."
    )


if __name__ == "__main__":
    main()