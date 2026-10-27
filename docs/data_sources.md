# Data Sources

## Project

Multi-Scale Forecasting and Transmission Modelling of Respiratory Virus-Associated Disease Burden

## Study Setting

Singapore

## Purpose of This Document

This document records the planned data sources for the project and the role of each dataset in the analysis.

The project will initially focus on publicly available respiratory surveillance data and later incorporate population contact structure for transmission modelling.

Clinical and airway microbiome data will be treated as a later-stage extension.

---

## 1. Singapore Respiratory Surveillance Data

### Primary purpose

Respiratory surveillance data will provide:

- the primary forecasting outcome
- recent respiratory disease activity
- co-circulating pathogen predictors
- the basis for short-term forecasting analyses

### Primary candidate outcome

The initial candidate primary outcome is:

- Average daily polyclinic attendance for Upper Respiratory Tract Infection (URTI), reported by epidemiological week

The exact outcome definition will be finalized after assessing:

- historical availability
- consistency across years
- missingness
- reporting changes
- epidemiological interpretation

### Planned source

Singapore Communicable Diseases Agency and/or Ministry of Health infectious-disease surveillance publications.

Potential sources include:

- Weekly Infectious Diseases Bulletins
- respiratory surveillance reports
- influenza surveillance publications
- other official surveillance datasets where appropriate

### Raw data location

Original source files will be stored in:

```text
data/raw/singapore_surveillance/
```

Raw files should not be manually edited.

Any cleaning or transformation will be performed through reproducible scripts.

---

## 2. Upper Respiratory Tract Infection Data

### Role in the project

URTI activity is the primary candidate outcome representing population-level respiratory disease burden.

### Candidate variable

- Average daily polyclinic attendance for URTI

### Desired temporal resolution

- Weekly

### Required fields

Where available:

- year
- epidemiological week
- reporting date
- URTI activity
- source file
- source definition

### Data-quality questions

The audit will assess:

- earliest available year
- latest available year
- missing weeks
- duplicate weeks
- definition changes
- changes in healthcare reporting
- COVID-era disruptions
- unusual structural breaks

---

## 3. Influenza Surveillance Data

### Role in the project

Influenza activity will be considered as a co-circulating respiratory-pathogen predictor.

### Candidate measures

Depending on availability:

- influenza test positivity
- influenza-like illness activity
- laboratory-confirmed influenza detections
- other routinely reported influenza surveillance indicators

### Important requirement

The exact influenza metric must be identified before modelling.

Different measures will not be treated as interchangeable.

For example:

```text
influenza positivity
```

is not automatically equivalent to:

```text
number of influenza cases
```

### Inclusion criteria

Influenza will be included in the main forecasting model only if:

- the weekly measure is clearly defined
- sufficient historical coverage is available
- reporting is reasonably consistent
- the series overlaps sufficiently with the URTI outcome

---

## 4. Respiratory Syncytial Virus (RSV) Data

### Role in the project

RSV activity will be considered as a candidate co-circulating respiratory-pathogen predictor.

### Candidate measures

Depending on availability:

- laboratory-confirmed RSV detections
- RSV positivity
- respiratory surveillance indicators involving RSV

### Data-quality questions

The source audit will determine:

- whether weekly RSV data are publicly available
- the length of the available series
- whether definitions changed over time
- whether there are large periods of missing data
- whether the series overlaps sufficiently with the URTI outcome

### Inclusion decision

RSV will not automatically be forced into the main forecasting model.

If historical coverage is limited, RSV may instead be included in:

- a shorter-period secondary analysis
- sensitivity analysis
- recent-period multi-pathogen modelling

---

## 5. SARS-CoV-2 Data

### Role in the project

SARS-CoV-2 activity will be considered as a candidate predictor for the recent-period respiratory forecasting analysis.

### Candidate measures

Potential measures may include:

- reported cases
- hospitalisations
- test positivity
- other official SARS-CoV-2 surveillance indicators

### Important limitation

SARS-CoV-2 surveillance begins much later than the likely URTI and influenza surveillance series.

The project will therefore not automatically restrict the entire study period to the COVID-19 era.

### Possible analysis strategy

A longer primary model may use:

```text
URTI + influenza
```

while a recent-period extended model may use:

```text
URTI + influenza + RSV + SARS-CoV-2
```

This decision will depend on the surveillance-source audit.

---

## 6. Epidemiological Time Variables

Temporal variables will be generated from the surveillance dataset rather than downloaded from an external source.

Potential variables include:

- epidemiological year
- epidemiological week
- calendar date
- annual seasonal terms
- sine seasonal term
- cosine seasonal term
- recent moving averages
- rolling trends

These variables will be generated reproducibly in the data-processing pipeline.

---

## 7. Lagged Epidemiological Predictors

Lagged versions of the primary outcome and pathogen surveillance variables will be generated during feature engineering.

Candidate lag intervals include:

- 1 week
- 2 weeks
- 3 weeks
- 4 weeks
- 6 weeks
- 8 weeks

The final lag structure will be selected using validation data only.

Future observations will never be used to create predictors for earlier forecast origins.

---

## 8. Population Contact Data

### Role in the project

Age-specific population contact data will be used during the transmission-modelling stage rather than as the main weekly forecasting input.

### Candidate source

Published synthetic contact matrices for Singapore.

### Candidate contact settings

Where available:

- home
- school
- workplace
- other locations
- total contacts

### Planned use

Contact matrices may be used to:

- construct age-structured transmission models
- represent population mixing patterns
- estimate transmission heterogeneity
- simulate intervention scenarios
- investigate targeted age-specific interventions

### Storage location

Planned raw contact data location:

```text
data/raw/contact_matrices/
```

---

## 9. Transmission Indicators

Transmission indicators may be estimated after the surveillance dataset has been cleaned and validated.

Potential derived variables include:

- time-varying reproduction number
- age-specific force of infection
- transmission intensity
- infection pressure

These quantities will not be treated as raw data sources.

They will be generated through the modelling pipeline and documented separately.

---

## 10. Airway Microbiome Data

### Project stage

Airway microbiome data will be treated as a later-stage extension.

The initial public-data forecasting analysis does not require microbiome data.

### Potential role

If suitable public or collaborator-provided data become available, microbiome features may be used to investigate whether airway microbial structure contributes additional information about respiratory disease severity or exacerbation risk.

### Candidate microbiome variables

Potential features may include:

- alpha diversity
- dominant microbial taxa
- relative abundances
- microbial community structure
- microbial dysbiosis measures
- pathogen presence
- microbiome clusters

### Important limitation

Public microbiome datasets will not be artificially linked to unrelated population surveillance observations.

A microbiome proof-of-concept analysis will remain separate unless genuinely linked patient-level data become available.

### Storage location

Planned raw microbiome data location:

```text
data/raw/microbiome/
```

or:

```text
microbiome/
```

depending on file size and processing requirements.

---

## 11. Patient-Level Clinical Data

Patient-level clinical data are not currently part of the public-data phase.

If suitable collaborator-provided data later become available, potential variables may include:

- age
- sex
- lung function
- bronchiectasis severity
- previous exacerbation history
- hospitalisation history
- antibiotic exposure
- microbiome characteristics
- respiratory-virus exposure
- exacerbation outcomes

These data would support a later clinical prediction model.

The current project will not claim to predict individual patient outcomes without appropriate linked clinical data.

---

## 12. Data Not Included in the Current Project

The current project design does not include climate or environmental data.

The following are therefore excluded from the primary modelling pipeline:

- ERA5 data
- temperature
- rainfall
- relative humidity
- absolute humidity
- dew-point temperature
- other meteorological predictors

This decision was made to maintain a focused and reproducible project centred on:

- respiratory surveillance
- cross-pathogen activity
- seasonality
- population contact structure
- transmission modelling
- intervention evaluation

---

## 13. Planned Data Directory Structure

```text
data/
├── raw/
│   ├── singapore_surveillance/
│   ├── contact_matrices/
│   └── microbiome/
│
├── interim/
│   ├── surveillance_source_audit.csv
│   └── intermediate_cleaned_files/
│
└── processed/
    ├── respiratory_master_weekly.csv
    ├── forecasting_features.csv
    └── transmission_inputs.csv
```

The exact files will be created as the project progresses.

---

## 14. Raw Data Policy

Raw source files will never be manually edited.

All transformations will be performed using reproducible scripts.

The workflow will follow:

```text
RAW DATA
    ↓
INTERIM DATA
    ↓
PROCESSED DATA
    ↓
MODELLING DATASET
```

This separation is intended to preserve provenance and reproducibility.

---

## 15. Data Provenance Requirements

For each downloaded dataset or source file, the project should record:

- source organisation
- dataset name
- source URL
- date accessed
- temporal coverage
- variable definitions
- original file name
- any known reporting limitations
- any changes in definitions over time

These details will be maintained in the relevant raw-data README files and/or the surveillance-source audit.

---

## 16. Study-Period Selection

The final study period will be selected only after the surveillance-source audit is complete.

The chosen period should provide:

- sufficient weekly observations
- a stable primary outcome
- adequate pathogen-predictor coverage
- acceptable levels of missingness
- consistent surveillance definitions

The study period will not be chosen simply to maximize the number of predictors.

---

## 17. Final Data-Selection Principle

A variable will be included only if its epidemiological meaning, temporal coverage, and reporting consistency are sufficient to support valid modelling.

The project will prioritize:

```text
data quality
>
number of predictors
```

A smaller, well-defined dataset will be preferred over a larger but inconsistent dataset.