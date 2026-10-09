# Healthcare Questionnaire Risk Analysis

**MTech project:** Healthcare data analytics and explainable early-stage diabetes risk prediction using machine learning, SQL and Power BI.

## Project workflow

1. Downloads the UCI Early Stage Diabetes Risk Prediction dataset (Dataset ID 529).
2. Normalizes the questionnaire data and writes a data quality summary, including missing values and duplicate predictors.
3. Stores cleaned data in SQLite and exports descriptive SQL summaries.
4. Trains Logistic Regression, Random Forest and XGBoost using preprocessing pipelines.
5. Tunes each model with stratified 5-fold cross-validation on the training partition.
6. Keeps identical questionnaire profiles together in group-aware cross-validation and the held-out test split, selects by training-only CV ROC-AUC, then reports test results.
7. Reports Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, Brier score and confusion-matrix counts.
8. Produces a bootstrap interval for test ROC-AUC, a calibration plot and exploratory age/gender subgroup metrics.
9. Generates feature-importance and SHAP explanation artifacts for the selected model.
10. Exports dataset-wide and interactive assessment results for Power BI.

## Dataset limitation

The UCI dataset contains questionnaire variables such as age, gender, polyuria, polydipsia, sudden weight loss, weakness, polyphagia, visual blurring, itching, irritability, delayed healing, partial paresis, muscle stiffness, alopecia and obesity. It does **not** contain BMI, blood pressure or glucose measurements. The model is an educational research prototype, not a clinical diagnostic tool. Do not use it to make patient-care decisions.

## Run the project

Open this folder as a Python project.

### Windows

Double-click `RUN_PROJECT.bat`, or run it from a PowerShell terminal opened in the parent workspace:

```powershell
& .\MTech_Healthcare_Disease_Risk_Project\RUN_PROJECT.bat
```

The batch file creates a virtual environment, installs the dependencies, runs data preparation, SQL analysis and model training, then opens the existing Power BI dashboard. Review it and press Enter in the terminal to open the Diabetes Health Check in a local browser. Fill in age, gender, each symptom, and the recipient's email in the form, then submit it to view the patient report. The project updates the CSV outputs and attempts to send the result email. Keep the terminal open while using the form; press Ctrl+C in the terminal when finished. The final experiment report is generated and opened after you stop the assessment server. Refresh Power BI after the assessment to load the latest dashboard data. The first run needs internet access to download the dataset.

### Optional email setup

Email is sent through Gmail SMTP using `smtp.gmail.com` on port `587` with STARTTLS. Each message includes the patient profile as a PDF attachment saved to `outputs/patient_report.pdf`. The sender defaults to `banalasupriya2003@gmail.com`. The app reads `SMTP_PASSWORD` from the local `.env` file; if it is missing, it asks for the password once at startup. The `.env` file is excluded from Git. The HTML and PDF reports are still created if email sending fails, and the result page displays the SMTP error.

The requested email mapping is used: a model `Negative` result gets a caution/follow-up message, and a `Positive` result gets a routine notification. Both messages explain that the result is not a diagnosis. A negative result cannot rule out diabetes; people with concerns should consult a healthcare professional. The recipient is entered in the browser form; Gmail SMTP sends through the configured Gmail account.

### Manual run from this project folder

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_project.py
```

To run only another patient assessment after training, use:

```powershell
.venv\Scripts\python.exe src\predict_patient.py
```

## Key outputs

- `outputs/data_quality_report.txt` — missing values, duplicate predictors and target distribution.
- `outputs/model_metrics.csv` — cross-validation ranking and held-out test metrics.
- `outputs/run_summary.json` — run configuration, package versions and selected model results.
- `outputs/experiment_report.md` — generated results, data quality, subgroup metrics and limitations for thesis review.
- `outputs/experiment_report.html` — browser version of the final experiment report, opened after the assessment ends.
- `outputs/test_predictions.csv` — predictions for held-out test rows.
- `outputs/subgroup_metrics.csv` — exploratory test metrics by age and gender; subgroup counts are small.
- `outputs/powerbi_patient_risk.csv` — dataset-wide model scores and labels.
- `outputs/prediction_result.csv` — the latest interactive assessment.
- `outputs/patient_report.html` — readable report for the latest assessment, with questionnaire answers and model result.
- `outputs/patient_report.pdf` — PDF patient profile attached to the result email.
- `outputs/dashboard_data.csv` — dataset-wide rows plus the latest assessment for Power BI.
- `outputs/feature_importance.csv` and `outputs/shap_global_importance.csv` — global model explanations.
- `outputs/roc_curve_comparison.png`, `outputs/confusion_matrix_best.png`, `outputs/calibration_curve_best.png` and `outputs/shap_global_bar.png` — evaluation figures.
- `outputs/healthcare.db` — SQLite database with the cleaned data and SQL analysis tables.

## Power BI

The supplied `powerbi file.pbix` currently contains saved report data and is not connected to the generated CSV files. In Power BI Desktop, use **Home → Get data → Text/CSV** to import `outputs/dashboard_data.csv`, then save the PBIX. After each assessment, use **Refresh** to load the updated file. Use `record_type` to distinguish dataset observations from the latest assessment; the assessment has no actual target label. See `docs/POWER_BI_GUIDE.md` for a simple one-page dashboard layout.

## Research methodology and reporting

See `docs/RESEARCH_METHODOLOGY.md` for the split and model-selection protocol, interpretation notes and limitations. Use the generated output files for reported numerical results; do not copy example metrics from older runs.

## Dataset citation

M. M. F. Islam et al., *Early Stage Diabetes Risk Prediction*, UCI Machine Learning Repository, Dataset 529. DOI: https://doi.org/10.24432/C5VG8H.
