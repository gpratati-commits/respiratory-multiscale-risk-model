# Forecast Outcome Definition

## Project

Multi-Scale Forecasting of Respiratory Virus-Associated Disease Burden

## Study Setting

Singapore

## Primary Forecast Outcome

Weekly respiratory disease burden represented by the average daily
polyclinic attendance for Upper Respiratory Tract Infection (URTI).

## Reason for Selection

URTI polyclinic attendance is routinely reported through Singapore's
infectious-disease surveillance system and provides a population-level
measure of respiratory disease burden.

It is suitable for short-term forecasting because it is repeatedly
measured over time and is directly relevant to healthcare resource use.

## Forecast Horizons

The primary forecasting horizons will be:

- 1 week ahead
- 2 weeks ahead
- 4 weeks ahead

## Candidate Predictors

### Epidemiological predictors

- Previous URTI activity
- Influenza activity
- RSV activity, if a sufficiently consistent historical series is available
- SARS-CoV-2 activity, if a sufficiently consistent historical series is available

### Environmental predictors

- Temperature
- Rainfall
- Relative humidity
- Absolute humidity

### Temporal predictors

- Epidemiological week
- Annual seasonality
- Lagged respiratory activity

### Later-stage predictors

- Age-specific social contact patterns
- Population transmission indicators
- Airway microbiome features

## Important Scope Decision

The initial public-data analysis will forecast population-level respiratory
disease burden.

It will NOT claim to predict bronchiectasis exacerbations or individual
patient outcomes.

Patient-level exacerbation prediction will be considered only if suitable
clinical and microbiome data become available.