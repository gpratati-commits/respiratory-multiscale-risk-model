# Forecast Outcome Definition

## Project

Multi-Scale Forecasting and Transmission Modelling of Respiratory Virus-Associated Disease Burden

## Study Setting

Singapore

## Primary Forecast Outcome

The primary forecasting outcome will be a weekly population-level measure of respiratory disease burden in Singapore.

The initial candidate outcome is:

- Average daily polyclinic attendance for Upper Respiratory Tract Infection (URTI), reported by epidemiological week.

The final outcome definition will be confirmed after assessing the consistency, duration, missingness, and reporting definitions of the available Singapore surveillance data.

## Primary Research Question

Can recent activity of co-circulating respiratory pathogens improve 1-, 2-, and 4-week-ahead prediction of respiratory disease burden in Singapore beyond models based on recent respiratory disease activity and seasonality alone?

## Secondary Research Question

How can age-specific population contact structure be incorporated into transmission models to evaluate respiratory-virus transmission and targeted intervention strategies?

## Forecast Horizons

The primary forecast horizons will be:

- 1 week ahead
- 2 weeks ahead
- 4 weeks ahead

## Core Forecasting Predictors

### 1. Recent respiratory disease activity

Lagged values of the primary respiratory outcome will be used to represent recent disease activity.

Candidate lags may include:

- 1 week
- 2 weeks
- 3 weeks
- 4 weeks
- 6 weeks
- 8 weeks

The final lag structure will be selected using validation data only.

### 2. Seasonal and temporal indicators

Candidate temporal predictors may include:

- Epidemiological week
- Annual seasonality
- Sine and cosine seasonal terms
- Calendar year
- Recent moving averages
- Recent trends

### 3. Co-circulating respiratory pathogen activity

Candidate pathogen predictors include:

- Influenza
- Respiratory syncytial virus (RSV)
- SARS-CoV-2

Lagged pathogen activity will be considered where sufficiently consistent weekly surveillance series are available.

A pathogen will not automatically be included in the primary model merely because data exist. Inclusion will depend on:

- temporal coverage
- consistency of surveillance definitions
- missingness
- overlap with the primary outcome
- epidemiological interpretability

Shorter or less consistent pathogen series may instead be used in sensitivity or secondary analyses.

## Predictor Comparison Strategy

The forecasting analysis will compare nested predictor sets.

### Model Set A: Outcome history

- Lagged respiratory disease activity only

### Model Set B: Outcome history plus seasonality

- Lagged respiratory disease activity
- Seasonal and temporal predictors

### Model Set C: Multi-pathogen model

- Lagged respiratory disease activity
- Seasonal and temporal predictors
- Lagged activity of co-circulating respiratory pathogens

The main forecasting question will be whether Model Set C improves out-of-sample performance compared with Model Set B.

## Planned Forecasting Models

### Baseline models

- Persistence forecast
- Recent moving-average forecast
- Seasonal naive forecast

### Statistical model

- Negative Binomial regression

### Machine-learning model

- XGBoost

Additional models may be considered only if they provide a clear methodological contribution.

## Validation Strategy

All forecasting evaluation will be chronological.

The data will be divided into:

- Training period
- Validation period
- Held-out test period

The validation period will be used for:

- hyperparameter selection
- predictor selection
- lag selection
- model selection

The held-out test period will remain untouched until the final evaluation.

No future observations will be used to construct predictors for earlier forecast origins.

## Evaluation Metrics

Planned deterministic forecast metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Scaled Error (MASE)

Probabilistic metrics may be added later if probabilistic forecasts are developed.

## Transmission-Modelling Extension

Age-specific population contact structure will be introduced after completion of the surveillance forecasting analysis.

Contact matrices will be used to support:

- age-structured transmission modelling
- estimation of transmission heterogeneity
- intervention simulations
- comparison of age-targeted strategies

Population contact structure will therefore be treated primarily as a transmission-modelling input rather than as a standard weekly forecasting predictor.

## Clinical and Microbiome Extension

Airway microbiome and patient-level clinical data will be treated as a later-stage extension.

Potential future predictors may include:

- airway microbial diversity
- dominant microbial taxa
- microbiome community structure
- age
- lung-function measures
- previous exacerbation history
- antibiotic exposure
- disease severity

The public-data phase of this project will not claim to predict individual bronchiectasis exacerbations or hospitalisation risk unless suitable linked patient-level data become available.

## Scope Decision

Climate and environmental predictors are not included in the current project design.

The primary focus is:

- respiratory surveillance
- cross-pathogen activity
- seasonality
- short-term forecasting
- population contact structure
- transmission modelling
- intervention evaluation

This narrower design is intended to maintain a clear and reproducible modelling framework while allowing later clinical and microbiome extensions.

## Final Analysis Principle

The final study period and predictor set will be determined after surveillance-source auditing.

The modelling design will not force inclusion of variables with inconsistent historical coverage solely to increase model complexity.