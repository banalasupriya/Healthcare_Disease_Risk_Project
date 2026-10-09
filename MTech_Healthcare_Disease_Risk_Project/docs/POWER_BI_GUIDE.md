# Power BI Dashboard

## Existing report pages

Keep the existing dashboard. It already has four useful pages: **Disease Risk Overview**, **Symptom Analysis**, **Model Performance**, and **SHAP Explainability**. Together they cover the cohort, questionnaire patterns, model evaluation, and model explanations.

## Connect it to the latest project data

The supplied `powerbi file.pbix` contains saved report data and is not connected to the generated CSV files yet.

1. Open `powerbi file.pbix` in Power BI Desktop.
2. Select **Home → Get data → Text/CSV** and choose `outputs/dashboard_data.csv`.
3. Select **Load**, then save the PBIX so it remembers the CSV connection.
4. After a new Diabetes Health Check is submitted, select **Refresh** to load the latest row.

## Read results carefully

- The interactive assessment has no actual `target` label. Do not compare it against actual outcomes.
- `risk_category` describes the model score; it is not a clinical diabetes level.
- `diabetes_level` is based on the entered fasting glucose value. It is separate from the questionnaire model prediction.
- The project CSV files are saved under `outputs/`. If Power BI does not refresh the latest values, check that its CSV source points to this project's `outputs/dashboard_data.csv`.
