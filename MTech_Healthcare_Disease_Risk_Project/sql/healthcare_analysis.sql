-- Run against outputs/healthcare.db
SELECT COUNT(*) AS total_patients, SUM(target) AS positive_patients, ROUND(100.0*AVG(target),2) AS positive_rate_pct FROM patient_health;
SELECT gender, COUNT(*) patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health GROUP BY gender;
SELECT CASE WHEN age<30 THEN '18-29' WHEN age<45 THEN '30-44' WHEN age<60 THEN '45-59' ELSE '60+' END age_group, COUNT(*) patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health GROUP BY age_group ORDER BY age_group;
SELECT obesity, COUNT(*) patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health GROUP BY obesity;
SELECT polyuria, polydipsia, COUNT(*) patients, ROUND(100.0*AVG(target),2) positive_rate_pct FROM patient_health GROUP BY polyuria, polydipsia ORDER BY positive_rate_pct DESC;
