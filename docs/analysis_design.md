# Analysis Design

## Primary Outcome

## Primary Forecasting Analysis

## Recent Multi-Pathogen Extension

## Influenza Surveillance Definition Change

## RSV Decision

## SARS-CoV-2 Decision

## Forecast Horizons

## Planned Models

## Validation Strategy
## ARI Quality-Control Review

The weekly ARI series was screened for missing observations,
duplicate epidemiological weeks, non-positive values, statistical
outliers, and unusually large week-to-week changes.

Three observations were flagged for manual review:

- 2020 E-week 5: ARI attendance = 4728
- 2020 E-week 6: ARI attendance = 3791
- 2022 E-week 5: ARI attendance = 2945

These observations were retained if their values matched the
corresponding original surveillance bulletins.

Flagged observations were not automatically removed because unusual
weekly respiratory activity may represent genuine epidemiological or
healthcare-utilisation changes rather than data errors.
## Study-Period Decision Following Exploratory Analysis

Exploratory analysis identified a major structural change in weekly
ARI polyclinic attendance during 2020-2022.

The 2019 series represented relatively high pre-pandemic respiratory
healthcare utilisation, followed by a sharp disruption during 2020
and persistently lower attendance during much of 2020-2021.
Attendance subsequently recovered during 2022.

From 2023 onward, the ARI series showed a substantially more stable
distribution and level of respiratory healthcare utilisation.

For this reason, the primary forecasting analysis will use the
recent, more stable surveillance period:

- Primary analysis period: 2023 E-week 1 through 2026 E-week 41
- Training period: 2023-2024
- Validation period: 2025
- Held-out test period: 2026 E-weeks 1-41

The full 2019-2026 series will be retained for sensitivity and
historical analyses rather than treated as a single homogeneous
forecasting regime.

## Multi-Pathogen Extension

The representative surveillance audit showed that weekly influenza
positivity became directly available in the newer surveillance format
from 2024 onward.

Therefore, the multi-pathogen extension will primarily use:

- 2024 through 2026 E-week 41

Potential predictors will include:

- lagged ARI activity
- influenza activity
- COVID-19 activity
- seasonality

RSV will be included only where a consistent and interpretable weekly
measure can be constructed.

## Interpretation of Full-Period Autocorrelation

The full-period ARI series showed very strong short-lag autocorrelation
with slow decay across longer lags.

Because the series contains the major 2020-2022 pandemic-era structural
disruption, this full-period autocorrelation should not be interpreted
as evidence of stationary seasonal dynamics.

Autocorrelation and seasonal structure will therefore also be evaluated
within the recent 2023-2026 forecasting period.
## Recent-Period ARI Dynamics

Diagnostics restricted to the recent surveillance period showed strong
short-term dependence in weekly ARI polyclinic attendance.

Autocorrelation was highest at lag 1 and declined rapidly over the first
several weeks, supporting the use of recent ARI history as a primary
forecasting predictor.

The autocorrelation function also showed a broader oscillating structure,
including negative correlation around approximately half a year and renewed
positive correlation near the annual lag. This pattern is consistent with
annual seasonal structure, although formal statistical significance is not
inferred from the descriptive autocorrelation plot alone.

The 2023-2025 seasonal profile showed recurrent variation across the year,
including relatively elevated ARI attendance during the early part of the
year, another elevated period around E-weeks 17-21, lower attendance around
E-weeks 23-25, and increasing activity toward the end of the year.

Because only three recent complete surveillance years are available,
seasonality will be represented parsimoniously using Fourier sine/cosine
terms rather than separate parameters for every epidemiological week.

### Candidate ARI Predictors

The primary forecasting feature set will initially include:

- current ARI activity at the forecast origin
- ARI activity during the previous 1-4 weeks
- annual Fourier seasonal terms
- optional second-harmonic seasonal terms
- previous-year same-E-week ARI as a candidate seasonal benchmark

Longer historical lags will not automatically be included solely because
of autocorrelation. Their forecasting value will be evaluated using the
chronological validation period.

### Chronological Forecast Evaluation

Forecast targets will be divided chronologically:

- Training targets: 2023-2024
- Validation targets: 2025
- Final held-out test targets: 2026 E-weeks 1-41

Forecast horizons will be 1, 2 and 4 weeks ahead.

All feature selection and hyperparameter decisions will use training and
validation data only. The 2026 test period will remain untouched until the
final model evaluation.
## Baseline Forecasting Results

Three simple forecasting benchmarks were evaluated at 1-, 2-, and
4-week horizons: persistence, a recent four-week mean, and a
seasonal-naive forecast based on the corresponding target
epidemiological week in the previous year.

Model-development decisions were based on the 2025 validation period.

At the 1-week horizon, persistence was the strongest baseline
(MAE = 116.19; RMSE = 149.37; MASE = 0.802), consistent with the
strong short-lag autocorrelation observed in recent ARI activity.

At the 2-week horizon, persistence remained the strongest baseline
(MAE = 196.40; RMSE = 249.87; MASE = 1.356), although performance
declined relative to the 1-week horizon.

At the 4-week horizon, the seasonal-naive forecast performed best
(MAE = 238.56; RMSE = 303.17; MASE = 1.647), outperforming both
persistence and the recent four-week mean. This suggests that annual
seasonal information becomes relatively more useful as the forecast
horizon increases.

The recent four-week mean did not outperform persistence at any
validation horizon.

These benchmarks will be used as reference models when evaluating
statistical and machine-learning approaches.

The 2026 period is retained as the final held-out test set and will
not be used for subsequent feature selection or hyperparameter tuning.
## Negative Binomial Forecasting Results

Negative Binomial regression models were evaluated using the 2025
validation period at 1-, 2-, and 4-week forecast horizons.

Across all three horizons, the strongest validation performance was
obtained using recent ARI activity combined with first-harmonic annual
seasonality. The selected predictor set consisted of:

- current ARI attendance
- ARI lag 1
- ARI lag 2
- first-harmonic sine seasonal term
- first-harmonic cosine seasonal term

Validation performance was:

- 1-week horizon: MAE = 98.52, RMSE = 132.99, MASE = 0.680
- 2-week horizon: MAE = 152.68, RMSE = 205.31, MASE = 1.054
- 4-week horizon: MAE = 192.44, RMSE = 245.23, MASE = 1.329

All selected models converged successfully.

Adding a second seasonal harmonic or the previous-year same-week ARI
value did not improve validation performance. Therefore, the simpler
recent-history plus first-harmonic seasonality specification was retained.

The 2026 held-out test period was not used for feature selection.
## XGBoost Validation Results

XGBoost models were trained using 2023-2024 data and selected using
the 2025 validation period. Hyperparameter selection was performed
separately for the 1-, 2-, and 4-week forecast horizons.

The selected XGBoost models achieved:

- 1-week horizon: MAE = 105.39, RMSE = 140.28, MASE = 0.728
- 2-week horizon: MAE = 138.51, RMSE = 185.94, MASE = 0.956
- 4-week horizon: MAE = 184.93, RMSE = 240.26, MASE = 1.277

The full recent-history plus seasonality feature set was selected at
all three horizons.

Compared with the Negative Binomial model, Negative Binomial provided
lower validation error at the 1-week horizon, whereas XGBoost provided
lower validation error at the 2- and 4-week horizons.

The 2026 held-out test period was not used for XGBoost model or
hyperparameter selection.