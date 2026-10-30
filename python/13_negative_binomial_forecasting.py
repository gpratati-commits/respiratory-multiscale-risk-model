"""
13_negative_binomial_forecasting.py

Fit Negative Binomial forecasting models for weekly ARI attendance.

IMPORTANT
---------
Only training data (2023-2024) are used for model fitting.

The 2025 validation period is used to compare candidate predictor sets.

The 2026 test period is NOT evaluated in this script.

Forecast horizons:
    1 week
    2 weeks
    4 weeks
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
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
    / "negative_binomial_validation_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "negative_binomial_validation_predictions.csv"
)


# Candidate predictor sets.
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

    "recent_seasonal_history": [
        "ari_current",
        "ari_lag_1",
        "ari_lag_2",
        "season_sin_1",
        "season_cos_1",
        "season_sin_2",
        "season_cos_2",
        "ari_target_same_week_previous_year",
    ],
}


def calculate_mae(y_true, y_pred):

    return np.mean(
        np.abs(y_true - y_pred)
    )


def calculate_rmse(y_true, y_pred):

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def standardize_train_validation(
    train,
    validation,
    features,
):

    train_x = train[
        features
    ].copy()

    validation_x = validation[
        features
    ].copy()

    means = train_x.mean()

    stds = train_x.std()

    # Prevent division by zero.
    stds = stds.replace(
        0,
        1.0,
    )

    train_scaled = (
        train_x - means
    ) / stds

    validation_scaled = (
        validation_x - means
    ) / stds

    return (
        train_scaled,
        validation_scaled,
    )


def main():

    print("=" * 70)
    print("NEGATIVE BINOMIAL ARI FORECASTING")
    print("=" * 70)

    forecast = pd.read_csv(
        INPUT_FILE
    )

    ari = pd.read_csv(
        ARI_FILE
    )

    # MASE denominator from training period only.
    train_ari = (
        ari[
            ari["year"].isin(
                [2023, 2024]
            )
        ]
        .sort_values(
            "time_index"
        )["ari_attendance"]
        .astype(float)
    )

    mase_scale = (
        train_ari
        .diff()
        .abs()
        .dropna()
        .mean()
    )

    print(
        f"\nTraining MASE scale: "
        f"{mase_scale:.3f}"
    )

    metrics_records = []
    prediction_records = []

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

            # Keep only rows with all required predictors.
            usable = horizon_data.dropna(
                subset=features
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

            (
                x_train,
                x_validation,
            ) = standardize_train_validation(
                train=train,
                validation=validation,
                features=features,
            )

            x_train = sm.add_constant(
                x_train,
                has_constant="add",
            )

            x_validation = sm.add_constant(
                x_validation,
                has_constant="add",
            )

            y_train = (
                train["target_ari"]
                .astype(float)
                .to_numpy()
            )

            y_validation = (
                validation["target_ari"]
                .astype(float)
                .to_numpy()
            )

            try:

                model = NegativeBinomial(
                    y_train,
                    x_train,
                    loglike_method="nb2",
                )

                result = model.fit(
                    disp=False,
                    maxiter=500,
                )

                prediction = np.asarray(
                    result.predict(
                        x_validation
                    )
                )

                # Guard against invalid numerical predictions.
                valid_prediction = (
                    np.isfinite(prediction)
                    &
                    (prediction >= 0)
                )

                y_true = (
                    y_validation[
                        valid_prediction
                    ]
                )

                y_pred = (
                    prediction[
                        valid_prediction
                    ]
                )

                if len(y_true) == 0:
                    continue

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

                metrics_records.append(
                    {
                        "horizon_weeks":
                            horizon,

                        "feature_set":
                            feature_set_name,

                        "n_train":
                            len(train),

                        "n_validation":
                            len(y_true),

                        "MAE":
                            model_mae,

                        "RMSE":
                            model_rmse,

                        "MASE":
                            model_mase,

                        "converged":
                            bool(
                                result.mle_retvals.get(
                                    "converged",
                                    False,
                                )
                            ),
                    }
                )

                validation_kept = (
                    validation
                    .iloc[
                        np.where(
                            valid_prediction
                        )[0]
                    ]
                    .copy()
                )

                for j in range(
                    len(validation_kept)
                ):

                    row = (
                        validation_kept
                        .iloc[j]
                    )

                    prediction_records.append(
                        {
                            "horizon_weeks":
                                horizon,

                            "feature_set":
                                feature_set_name,

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
                                    y_true[j]
                                ),

                            "predicted_ari":
                                float(
                                    y_pred[j]
                                ),
                        }
                    )

            except Exception as exc:

                print(
                    f"FAILED: "
                    f"{feature_set_name} "
                    f"at horizon {horizon}: "
                    f"{exc}"
                )

    metrics = pd.DataFrame(
        metrics_records
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

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS — NEGATIVE BINOMIAL")
    print("=" * 70)

    print(
        metrics
        .round(3)
        .to_string(
            index=False
        )
    )

    print("\nBEST FEATURE SET BY HORIZON")

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

    print(
        best[
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

    print(
        f"\nMetrics saved to:\n"
        f"{METRICS_FILE}"
    )

    print(
        f"\nPredictions saved to:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        "\nIMPORTANT: "
        "The 2026 test set was not evaluated."
    )


if __name__ == "__main__":
    main()