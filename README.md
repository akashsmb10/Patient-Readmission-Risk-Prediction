# Patient Readmission Risk Prediction

Academic portfolio study only. This is not a clinical tool and must not be used for treatment, triage, discharge, or deployment decisions.

## Data and target

Source: [UCI Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008), CC BY 4.0. It contains de-identified diabetic inpatient encounters from 130 US hospitals/integrated networks. The prediction point is **at discharge**. Target `1` means the dataset label is `<30`; target `0` combines `NO` and `>30`.

Features are restricted to discharge-available encounter fields. Encounter ID, patient ID, target, race, and gender are excluded from model features. Race and gender are retained only for a descriptive subgroup audit. Expired/hospice discharge dispositions are excluded because their endpoint differs from readmission.

## Protocol

Patients, rather than encounters, are disjoint across train, calibration, validation, and test sets. `Pipeline` and `ColumnTransformer` fit imputation, scaling, and encoding on training data only. The study compares DummyClassifier, Logistic Regression, Random Forest, and Gradient Boosting. Model selection uses validation average precision; test results are held out until selection. Calibration is fit on the calibration set.

The threshold maximizes validation F2 across 0.02–0.60. F2 weights recall more than precision as an explicit academic demonstration of prioritizing missed positive cases; it is not a clinical utility policy. Metrics include precision, recall, F1, ROC-AUC, PR-AUC (average precision), calibration/Brier score, and a confusion matrix.

## Run

```powershell
cd patient-readmission-risk-prediction
.\.venv-p2\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py --bootstrap 100
python -m pytest -q
```

Outputs include validation and test comparison tables, evaluation chart, split manifest, subgroup audit, permutation importance, SHAP feature importance, five prediction examples, model artifact, and run manifest.

## Limitations and fairness

This historical benchmark lacks reliable event dates and hospital identifiers, so it is neither temporal nor external validation. Repeated encounters within a split remain correlated. Dataset labels may miss readmissions outside the recorded system. Subgroup metrics are exploratory, omit small groups, and do not prove fairness. SHAP and permutation importance describe model reliance, not causal effects. The data contain sensitive demographics and should not be used to automate decisions about people.

See [model card](reports/model_card.md) for intended use, risks, privacy, and monitoring limits.
