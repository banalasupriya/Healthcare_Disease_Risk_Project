# Research Methodology and Evaluation Notes

## Research question

Can questionnaire variables in the UCI Early Stage Diabetes Risk Prediction dataset distinguish the dataset's recorded positive and negative classes? This project evaluates that question as a small educational classification study. It does not establish clinical utility or causal relationships.

## Data and target

The project downloads UCI Machine Learning Repository Dataset 529. The target is normalized to `0 = negative` and `1 = positive`. Predictors are the dataset's age, gender and symptom/questionnaire fields. The source does not contain glucose, BMI or blood-pressure measurements. `data_quality_report.txt` records row counts, class distribution, missing values and repeated predictor patterns for each run.

## Preprocessing

Numeric predictors are median-imputed and standardized. Categorical predictors are imputed with the most frequent training value and one-hot encoded. The transformations are inside a scikit-learn pipeline, so each cross-validation fold learns preprocessing parameters only from that fold's training portion. Unknown categories at prediction time are ignored by the one-hot encoder.

## Model development

Exact duplicate predictor profiles are assigned the same group. A stratified 5-fold group split with random seed 42 reserves one fold (approximately 20% of rows) as the held-out test partition. Logistic Regression, Random Forest and XGBoost are tuned with stratified 5-fold group cross-validation on the remaining training partition. The model with the highest mean cross-validation ROC-AUC is selected. The test partition is kept out of tuning and model selection, and is used to report final comparison metrics. Grouping prevents identical questionnaire profiles from appearing on both sides of a split.

Classification labels use a fixed 0.50 score threshold. This is a project convention, not a clinically optimized decision threshold. The model probability score is not automatically calibrated. The test Brier score and calibration plot provide calibration diagnostics, but do not make the score suitable for clinical decisions.

## Evaluation artifacts

- `model_metrics.csv`: mean and standard deviation of cross-validation ROC-AUC plus held-out test metrics.
- `run_summary.json`: selected model, random seed, split sizes, software versions, test ROC-AUC and a bootstrap interval.
- `test_predictions.csv`: actual labels, predicted labels and scores for held-out rows.
- `subgroup_metrics.csv`: exploratory test metrics by age group and gender, with sample counts.
- `roc_curve_comparison.png`, `confusion_matrix_best.png`, `calibration_curve_best.png`: test-set evaluation plots.
- `feature_importance.csv`, `shap_global_importance.csv`, `shap_global_bar.png`: global explanations for the selected model.

The ROC-AUC interval is a percentile bootstrap over distinct predictor-profile groups in the held-out test set. It reflects test-sample uncertainty only; it does not account for dataset shift, hyperparameter search uncertainty or external validation. Small subgroup counts make those metrics descriptive and unstable.

## Data leakage and duplicate observations

The data quality report flags duplicate predictor rows and duplicate predictor groups with conflicting target labels. Identical predictor profiles are kept together by the group-aware split. If identical profiles have conflicting target labels, investigate the source records before interpreting results. The dataset does not provide person identifiers, so grouping repeated profiles cannot prove that different rows belong to the same or different people.

## Explainability and responsible interpretation

Feature importance and SHAP describe how the selected model uses the questionnaire features. They do not show that a feature causes disease, that a feature is clinically useful, or that the model generalizes to another population. The source dataset is small and geographically specific. External validation on a suitable independent cohort would be required before any real-world use.

## Power BI

`powerbi_patient_risk.csv` includes recorded cohort labels, model scores and class predictions. The `evaluation_partition` column marks in-sample training scores separately from held-out test scores. Use only held-out rows for evaluation visuals. `dashboard_data.csv` appends the latest interactive questionnaire assessment with a blank actual target; keep it out of visuals comparing predictions with known outcomes. The supplied PBIX has no connection to generated files, so import the current CSV explicitly in Power BI Desktop.
