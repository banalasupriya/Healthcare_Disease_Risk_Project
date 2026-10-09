# Healthcare Disease Risk Prediction Using Machine Learning

## MTech Final Project Report

### A Data Science Approach Using Python, SQL, Power BI, and Explainable AI

**Student Name:** __________________________

**Roll Number:** __________________________

**Programme:** Master of Technology (MTech)

**Department:** __________________________

**College/University:** __________________________

**Academic Year:** 2026–2027

---

# Abstract

Healthcare organizations generate large amounts of patient-related data that can be analyzed to support early disease-risk assessment. This project presents a machine learning based approach for predicting diabetes risk using patient demographic and symptom information.

The study uses the UCI Early Stage Diabetes Risk Prediction dataset containing 520 patient records and symptom-related attributes. The data is cleaned and transformed using Python before being analyzed using SQL for healthcare-oriented analytical queries. Three machine learning algorithms are implemented and compared: Logistic Regression, Random Forest, and XGBoost.

The models are evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC. Explainable Artificial Intelligence is incorporated using SHAP (SHapley Additive exPlanations) to identify the features that have the greatest influence on model predictions. A Power BI dashboard is also developed to provide interactive visualization of patient statistics, disease-risk patterns, symptom analysis, model performance, and SHAP feature importance.

The project demonstrates an end-to-end healthcare data science workflow covering data preprocessing, SQL analytics, machine learning, model evaluation, explainability, and business intelligence visualization. The developed system is intended as a decision-support and academic analytical tool rather than a clinical diagnostic system.

---

# Chapter 1: Introduction

## 1.1 Background

Diabetes is one of the major chronic diseases affecting populations worldwide. Early identification of individuals who may be at higher risk can support timely medical evaluation and preventive healthcare.

With the growth of healthcare data, machine learning techniques can be used to identify patterns within patient information. Machine learning models can analyze multiple demographic and symptom-related variables simultaneously and generate predictions that may assist healthcare professionals in decision-support environments.

This project focuses on developing a complete healthcare data science pipeline for diabetes risk prediction. The pipeline includes data collection, preprocessing, exploratory analysis, SQL-based analytics, machine learning model development, model evaluation, explainable artificial intelligence, and Power BI visualization.

## 1.2 Problem Statement

Traditional healthcare assessment may require medical professionals to manually examine several patient characteristics and symptoms before determining whether further diabetes-related evaluation is appropriate.

The objective of this project is to develop a data-driven system that can analyze patient information and predict diabetes risk using machine learning algorithms. The system should also provide understandable analytical results through visualization and explainability techniques.

## 1.3 Objectives

The main objectives of this project are:

1. To obtain and prepare a healthcare diabetes-risk dataset.
2. To clean and transform the patient data using Python.
3. To perform healthcare-oriented analysis using SQL.
4. To develop diabetes-risk prediction models using:
   - Logistic Regression
   - Random Forest
   - XGBoost
5. To compare the models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
6. To identify important prediction features using SHAP explainability.
7. To develop an interactive Power BI dashboard.
8. To provide an end-to-end data science solution suitable for academic healthcare analytics.

## 1.4 Scope of the Project

The project focuses on diabetes risk prediction using demographic and symptom-related information contained in the selected dataset.

The project covers:

- Data preprocessing
- Exploratory data analysis
- SQL analytics
- Machine learning
- Model evaluation
- Explainable AI
- Power BI dashboard development

The system is designed for academic and decision-support purposes. It should not be considered a replacement for professional medical diagnosis or clinical assessment.

## 1.5 Technologies Used

The major technologies used in this project are:

| Technology | Purpose |
|---|---|
| Python | Data preprocessing and machine learning |
| Pandas | Data manipulation |
| NumPy | Numerical processing |
| Scikit-learn | Machine learning and evaluation |
| XGBoost | Gradient boosting model |
| SHAP | Model explainability |
| SQL / SQLite | Healthcare data analysis |
| Power BI | Interactive dashboard |
| Joblib | Saving trained machine learning models |
| Visual Studio Code | Development environment |

---

# Chapter 2: Dataset and Data Source

## 2.1 Dataset

The project uses the UCI Early Stage Diabetes Risk Prediction dataset.

The dataset contains:

- 520 patient records
- Demographic information
- Diabetes-related symptoms
- A classification target indicating positive or negative diabetes status

The target variable is converted into a numerical representation:

- `1` = Positive
- `0` = Negative

## 2.2 Dataset Features

Important features include:

- Age
- Gender
- Polyuria
- Polydipsia
- Sudden weight loss
- Weakness
- Polyphagia
- Genital thrush
- Visual blurring
- Itching
- Irritability
- Delayed healing
- Partial paresis
- Muscle stiffness
- Alopecia
- Obesity

## 2.3 Dataset Limitation

The selected dataset is symptom-oriented. It does not contain direct measurements such as BMI, blood pressure, or blood glucose level.

Therefore, the project does not claim to predict diabetes using those measurements.

---

# Chapter 3: Data Preprocessing

## 3.1 Data Cleaning

The raw dataset was processed using Python and Pandas.

The preprocessing steps included:

1. Loading the raw CSV dataset.
2. Standardizing column names.
3. Removing unnecessary spaces from categorical values.
4. Converting categorical values into consistent formats.
5. Identifying the target column.
6. Converting the target variable into binary numerical values.
7. Generating a unique patient identifier.
8. Saving the cleaned dataset as `cleaned_diabetes.csv`.

## 3.2 Cleaned Dataset

The resulting cleaned dataset contains 520 records and 18 columns, including the generated `patient_id` and numerical `target` variable.

The cleaned dataset is used by the SQL analysis, machine learning pipeline, and Power BI dashboard.

## 3.3 Target Distribution

The dataset contains:

- Positive cases: 320
- Negative cases: 200
- Total patients: 520

Therefore, positive cases represent approximately 61.54% of the dataset, while negative cases represent approximately 38.46%.

---

# Chapter 4: SQL Healthcare Analytics

## 4.1 SQL Database

The cleaned healthcare dataset was loaded into a SQLite database.

The generated database is:

`outputs/healthcare.db`

SQL queries were developed to analyze healthcare risk patterns.

## 4.2 SQL Analysis

The project performs analysis such as:

- Overall positive and negative case distribution
- Risk distribution by gender
- Risk distribution by age group
- Risk distribution by obesity
- Symptom-based risk analysis

The SQL analysis provides an additional analytical layer before machine learning modelling.

---

# Chapter 5: Machine Learning Methodology

## 5.1 Machine Learning Workflow

The machine learning workflow consists of:

1. Loading the cleaned dataset.
2. Separating features and target.
3. Encoding categorical variables.
4. Splitting the data into training and testing sets.
5. Training multiple machine learning algorithms.
6. Evaluating model performance.
7. Saving the trained models.
8. Selecting the best-performing model based on evaluation results.

## 5.2 Logistic Regression

Logistic Regression is used as a baseline classification algorithm. It estimates the probability that a patient belongs to the positive or negative class.

## 5.3 Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees. It can capture nonlinear relationships between patient symptoms and diabetes risk.

## 5.4 XGBoost

XGBoost is a gradient boosting algorithm that builds a sequence of decision trees to improve classification performance.

## 5.5 Model Evaluation Metrics

The following metrics are used:

### Accuracy

Accuracy measures the proportion of correctly classified observations.

### Precision

Precision measures the proportion of predicted positive cases that are actually positive.

### Recall

Recall measures the proportion of actual positive cases that are correctly identified.

### F1-Score

F1-score combines precision and recall into a single metric.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between positive and negative classes across different classification thresholds.

---

# Chapter 6: Model Results

The trained models produced the following test-set results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 98.08% | 100.00% | 96.88% | 98.41% | 99.96% |
| Random Forest | 98.08% | 98.44% | 98.44% | 98.44% | 99.84% |
| Logistic Regression | 93.27% | 98.31% | 90.63% | 94.31% | 99.06% |

XGBoost achieved the highest ROC-AUC and precision, while Random Forest achieved the highest recall and F1-score.

For a healthcare screening context, recall is particularly important because failing to identify a positive case can be undesirable. Therefore, Random Forest provides a strong alternative when prioritizing sensitivity, while XGBoost provides excellent overall discrimination.

---

# Chapter 7: Explainable AI Using SHAP

SHAP was used to improve the interpretability of the machine learning model.

The SHAP analysis identified several influential features, including:

- Polyuria
- Polydipsia
- Gender
- Itching
- Age
- Irritability
- Polyphagia
- Sudden weight loss
- Alopecia
- Genital thrush

SHAP feature importance indicates how strongly features contribute to model predictions. It does not mean that a feature directly causes diabetes.

The SHAP analysis is visualized in the Power BI dashboard.

---

# Chapter 8: Power BI Dashboard

The Power BI dashboard provides an interactive visual representation of the project results.

The dashboard contains the following pages:

1. Disease Risk Overview
2. Symptom Analysis
3. Model Performance
4. SHAP Explainability

## 8.1 Disease Risk Overview

The overview page contains key performance indicators including:

- Total Patients: 520
- Positive Cases: 320
- Negative Cases: 200

It also includes visualizations for:

- Positive and negative cases
- Diabetes risk by age group
- Diabetes cases by gender
- Diabetes risk by obesity

## 8.2 Symptom Analysis

The symptom analysis page visualizes positive cases according to different symptoms, including:

- Polyuria
- Polydipsia
- Sudden weight loss
- Weakness
- Polyphagia
- Itching
- Visual blurring
- Irritability
- Alopecia

## 8.3 Model Performance

The model performance page compares:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

for Logistic Regression, Random Forest, and XGBoost.

## 8.4 SHAP Explainability

The SHAP page displays the relative importance of model features and helps users understand which features have the greatest influence on model predictions.

---

# Chapter 9: Results and Discussion

The developed system successfully integrates data preprocessing, SQL analytics, machine learning, explainability, and business intelligence visualization.

The results demonstrate that all three machine learning models achieved strong classification performance on the selected dataset. XGBoost achieved the highest ROC-AUC of 99.96%, while Random Forest achieved the highest recall and F1-score among the evaluated models.

The Power BI dashboard provides an accessible way to explore patient distributions and model results. SHAP further improves transparency by showing which features contribute most strongly to predictions.

The results should be interpreted within the limitations of the dataset and should not be treated as evidence of clinical diagnostic performance.

---

# Chapter 10: Limitations

The project has several limitations:

1. The dataset contains only 520 patient records.
2. The dataset is symptom-oriented.
3. Important clinical measurements such as blood glucose, BMI, and blood pressure are not included.
4. The model performance may not generalize to other populations.
5. The system is not clinically validated.
6. The predictions should not be used as a substitute for professional medical diagnosis.

---

# Chapter 11: Conclusion and Future Work

## 11.1 Conclusion

This project developed an end-to-end healthcare disease-risk prediction system using Python, SQL, machine learning, SHAP explainability, and Power BI.

Three machine learning algorithms were implemented and compared. XGBoost and Random Forest demonstrated particularly strong performance on the selected dataset. The project also demonstrated how explainable AI can be integrated into healthcare analytics to make machine learning results easier to understand.

The Power BI dashboard provides an interactive interface for exploring patient statistics, symptom patterns, model performance, and feature importance.

Overall, the project demonstrates the practical application of data science techniques to healthcare risk analytics while emphasizing that machine learning predictions should support, rather than replace, professional medical decision-making.

## 11.2 Future Work

Future improvements could include:

- Using larger and more diverse healthcare datasets.
- Including clinical measurements such as blood glucose and BMI.
- Testing additional machine learning and deep learning algorithms.
- Performing external validation on independent datasets.
- Developing a web-based prediction interface.
- Adding real-time healthcare data integration.
- Improving fairness and bias analysis.
- Conducting clinical validation before any real-world deployment.

---

# References

1. UCI Machine Learning Repository, Early Stage Diabetes Risk Prediction Dataset.
2. Scikit-learn documentation.
3. XGBoost documentation.
4. SHAP documentation.
5. Microsoft Power BI documentation.
6. Python and Pandas documentation.
7. SQLite documentation.