# Data dictionary and feature policy

Source data are UCI Diabetes 130-US Hospitals for Years 1999–2008, licensed CC BY 4.0. Each row is a de-identified inpatient diabetic encounter.

| Group | Fields | Policy at discharge |
|---|---|---|
| Identifiers | `encounter_id`, `patient_nbr` | Used only for uniqueness and patient-disjoint splits; excluded from features |
| Target | `readmitted` | `<30` becomes 1; `NO` and `>30` become 0; excluded from features |
| Numeric features | time in hospital, lab procedures, procedures, medications, outpatient/emergency/inpatient visits, diagnoses | Median imputation and scaling inside the training-fitted pipeline |
| Categorical features | age, admission/discharge/source IDs, diagnoses, lab results, metformin, insulin, medication change, diabetes medication | Constant missing-value imputation and one-hot encoding inside the training-fitted pipeline |
| Sensitive fields | race, gender | Excluded from model features; retained only for exploratory subgroup reporting |

`?` is treated as missing. `None` for a lab result is retained as a meaningful observed category. Discharge disposition IDs 11, 13, 14, 19, 20, and 21 are excluded because the endpoint is not comparable to ordinary readmission. The dataset has no reliable dates for temporal validation and no hospital identifier for site validation.
