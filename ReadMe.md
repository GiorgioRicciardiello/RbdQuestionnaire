# A Two-Stage Questionnaire and Actigraphy Protocol for Remote Detection of Isolated REM Sleep Behavior Disorder

## Overview

This repository contains the code and analysis pipeline for the study:

**"A Two-Stage Questionnaire and Actigraphy Protocol for Remote Detection of Isolated REM Sleep Behavior Disorder in a Multicenter Cohort."**

The project integrates:

1. **A 4-item screening questionnaire** (RBD symptoms, hyposmia, constipation, orthostatic symptoms).
2. **Actigraphy-derived sleep features** extracted from wrist-worn AX6/AX3 accelerometers.
3. **A machine learning pipeline** using nested cross-validation and hyperparameter optimization (Optuna) to train classifiers.
4. **A two-stage screening protocol**: Stage 1 questionnaire → Stage 2 actigraphy confirmation.

---

## Study Cohorts

* **Mount Sinai Sleep and Healthy Aging Study (SHAS)**: Questionnaire + wearable data (n=62).
* **VascBrain cohort**: Wearable data (n=25).
* **Stanford Sleep Center**: Questionnaire + wearable data (n=84; 63 clinic, 21 community).
* **Stanford ADRC**: Wearable controls (n=78).
* **Mount Sinai sleep clinics**: Questionnaire only (n=147).

Inclusion: Age 40–80, no overt neurodegenerative disease. All iRBD cases were PSG-confirmed; controls with RBD-like symptoms but negative PSG were considered “mimics.”

---

## Methods

### Questionnaire

* **4 items**: RBD, hyposmia, constipation, orthostasis.
* Responses: *No = 0, Don’t know = 0.5, Yes = 1*.
* Models tested: Random Forest, LightGBM, XGBoost, Elastic Net.

### Actigraphy

* Devices: **Axivity AX6 (50 Hz, ±8g)** and **AX3 (100 Hz)**.
* Nights recorded: **6,620 nights** across 78 iRBD and 158 controls.
* Features extracted (n=113):

### Machine Learning Pipeline

* **Nested cross-validation**:

  * Outer loop: 10 folds (performance estimation).
  * Inner loop: 5 folds (Optuna hyperparameter tuning).
* **Models**: XGBoost for actigraphy, multiple classifiers for questionnaire.
* **Thresholds**:

  * τ = 0.5 (default).
  * τ* = Youden’s J (balanced Se/Sp).
  * Custom τ for maximizing Se (questionnaire) or Sp (actigraphy).
* Implemented in **Python 3.10** with scikit-learn, XGBoost, Optuna, statsmodels.

### Two-Stage Screening

* Stage 1: Questionnaire (maximize sensitivity).
* Stage 2: Actigraphy (maximize specificity).
* Final rule: classified as iRBD if **both** stages positive.

---

## Running the Pipelines

### Questionnaire Model

```bash
bsub < run_ml_questionnaire.sh
```

or locally:

```bash
python ml_questionnaire.py
```

### Actigraphy Model

```bash
bsub < run_ml_actigraphy.sh
```

or locally:

```bash
python ml_actigraphy.py
```

### Two-Stage Protocol

```bash
python two_stage_predictions.py
```

---

## Outputs

* **Metrics**:

  * `metrics/metrics_outer_folds_ci.csv` – performance with 95% CI.
* **Predictions**:

  * `predictions/predictions_outer_folds.csv` – subject-level predictions.
* **Models**:

  * Saved `.pkl` files per fold in `models/`.
* **Figures**:

  * ROC curves, confusion matrices, feature importance plots.

---

## Key Features

* Reproducible **nested CV** with group-stratified folds (subject-level).
* Configurable thresholds for clinical screening needs.
* Two-stage diagnostic mimicry (questionnaire + wearable).
* Confidence intervals for robust statistical reporting.
