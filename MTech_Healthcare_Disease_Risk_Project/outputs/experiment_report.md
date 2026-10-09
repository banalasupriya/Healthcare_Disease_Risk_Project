# Experiment Results

Generated at (UTC): 2026-10-08T13:10:48.355395+00:00

## Run configuration

- Selected model: **Random_Forest**
- Selection rule: mean stratified 5-fold CV ROC-AUC on training data
- Split: StratifiedGroupKFold by identical predictor profile
- Rows: 520 total; 416 training; 104 held-out test
- Unique predictor profiles: 251
- Held-out test profiles: 49
- Target positive rate: 61.5%
- Classification threshold: 0.50

## Model comparison

Models are ranked by training-only cross-validation ROC-AUC. Test metrics are reported on the held-out partition.

| Model | CV_ROC_AUC_Mean | Accuracy | Precision | Recall | F1 | ROC_AUC | Specificity | Brier_Score |
|---|---|---|---|---|---|---|---|---|
| Random_Forest | 0.969 | 0.933 | 0.983 | 0.906 | 0.943 | 0.973 | 0.975 | 0.066 |
| XGBoost | 0.965 | 0.923 | 0.983 | 0.891 | 0.934 | 0.970 | 0.975 | 0.069 |
| Logistic_Regression | 0.947 | 0.846 | 1.000 | 0.750 | 0.857 | 0.971 | 1.000 | 0.091 |

Selected model test ROC-AUC 95% profile-bootstrap interval: 0.8775740262321144 to 1.0.
Selected model test Brier score: 0.066.

## Data quality

```text
DATA QUALITY SUMMARY
====================
Source file: diabetes_data_upload.csv
Rows: 520
Predictor columns: 16
Missing predictor/target cells: 0
Rows with at least one missing value: 0
Duplicate predictor rows: 269
Duplicate predictor groups with conflicting targets: 0
Age range: 16 to 90; missing/non-numeric ages: 0.
Target distribution:
  Negative: 200 (38.5%)
  Positive: 320 (61.5%)

Duplicate patient records can make random train/test metrics optimistic;
interpret the scores with this limitation in mind.
```

## Exploratory subgroup metrics

These test-set subgroup results are descriptive. Include group sizes and avoid treating small groups as reliable fairness estimates.

| Group_Type | Group | Patients | Actual_Positive | Recall | Specificity | ROC_AUC |
|---|---|---|---|---|---|---|
| Gender | Female | 35 | 29 | 0.828 | 1.000 | 0.885 |
| Gender | Male | 69 | 35 | 0.971 | 0.971 | 0.992 |
| Age_Group | 30-44 | 44 | 22 | 0.773 | 1.000 | 1.000 |
| Age_Group | 45-59 | 42 | 28 | 0.964 | 0.929 | 0.972 |
| Age_Group | 60+ | 18 | 14 | 1.000 | 1.000 | 1.000 |

## Interpretation and limitations

This project analyzes a small public questionnaire dataset. Identical predictor profiles are grouped across splits, but the source has no person identifier, so profile groups cannot be confirmed as repeated or distinct people. The model score is not clinically calibrated or validated, and the model is not for diagnosis or patient-care decisions. External validation is needed before any real-world use.

The model explanations describe fitted model behavior; they do not show causation. The bootstrap interval reflects held-out profile-sample uncertainty and excludes dataset shift and model-selection uncertainty.
