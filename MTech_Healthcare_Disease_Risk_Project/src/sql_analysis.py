from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned_diabetes.csv"
DB = ROOT / "outputs" / "healthcare.db"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
with sqlite3.connect(DB) as con:
    df.to_sql("patient_health", con, if_exists="replace", index=False)
    queries = {
        "overall_risk.csv": "SELECT COUNT(*) total_patients, SUM(target) positive_patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health",
        "risk_by_gender.csv": "SELECT gender, COUNT(*) patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health GROUP BY gender",
        "risk_by_age_group.csv": "SELECT CASE WHEN age<30 THEN '16-29' WHEN age<45 THEN '30-44' WHEN age<60 THEN '45-59' ELSE '60+' END age_group, COUNT(*) patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health GROUP BY age_group ORDER BY age_group",
        "risk_by_obesity.csv": "SELECT obesity, COUNT(*) patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health GROUP BY obesity",
        "symptom_risk_summary.csv": "SELECT polyuria, polydipsia, sudden_weight_loss, weakness, COUNT(*) patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health GROUP BY polyuria, polydipsia, sudden_weight_loss, weakness ORDER BY positive_rate_pct DESC"
    }
    for filename, q in queries.items():
        pd.read_sql_query(q, con).to_csv(OUT / filename, index=False)
print(f"SQLite database created: {DB}")
