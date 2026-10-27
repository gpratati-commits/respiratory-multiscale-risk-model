# Data Sources

## Project

Multi-Scale Forecasting of Respiratory Virus-Associated Disease Burden

## Initial Study Setting

Singapore

## Primary Research Question

Can recent activity of influenza, RSV and SARS-CoV-2, together with
seasonality and environmental predictors, improve 1-, 2-, and
4-week-ahead prediction of respiratory disease burden compared with
models based only on recent outcome history?

---

## 1. Primary Respiratory Disease Outcome

Potential outcome:

Weekly respiratory disease burden in Singapore.

Candidate source:

Singapore Ministry of Health respiratory / infectious disease
surveillance data.

Variables to investigate:

- Epidemiological year
- Epidemiological week
- URTI activity or attendance
- Influenza activity
- RSV activity
- SARS-CoV-2 activity

The exact primary outcome will be finalized only after assessing
availability, consistency, missingness and changes in surveillance
definitions.

---

## 2. Environmental Data

Planned source:

ERA5-Land

Candidate variables:

- Mean temperature
- Minimum temperature
- Maximum temperature
- Dew-point temperature
- Relative humidity
- Absolute humidity
- Total precipitation

Environmental data will eventually be aggregated to epidemiological
weeks.

---

## 3. Social Contact Data

Planned source:

Published synthetic contact matrices.

Potential variables:

- Age-specific contact rates
- Home contacts
- Work contacts
- School contacts
- Other contacts

These will be used in later transmission-modelling analyses.

---

## 4. Airway Microbiome Data

Microbiome data will be treated as a later extension.

The first stage of the project will not claim to predict
bronchiectasis exacerbations because linked patient-level clinical
and microbiome data are not currently available.

A separate public microbiome proof-of-concept analysis may be added
after completion of the epidemiological forecasting component.

---

## Forecast Horizons

- 1 week ahead
- 2 weeks ahead
- 4 weeks ahead

## Planned Initial Models

### Baselines

- Persistence
- Moving-average baseline
- Seasonal naive baseline

### Statistical Model

- Negative Binomial regression

### Machine-Learning Model

- XGBoost

## Validation Strategy

All evaluation will be chronological:

TRAIN -> VALIDATION -> TEST

The validation period will be used for:

- Model selection
- Hyperparameter tuning
- Predictor selection

The test period will remain untouched until final evaluation.