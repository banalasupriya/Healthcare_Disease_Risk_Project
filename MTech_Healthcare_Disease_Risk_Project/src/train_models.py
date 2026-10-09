"""Train and evaluate the project classifiers with a held-out test set."""

from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "cleaned_diabetes.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50


def make_pipeline(model, numeric_columns, categorical_columns):
    """Create preprocessing and classifier steps fitted within each CV fold."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ]
    )
    return Pipeline([("preprocessor", preprocessing), ("model", model)])


def binary_metrics(y_true, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC_AUC": (
            roc_auc_score(y_true, probability)
            if np.unique(np.asarray(y_true)).size == 2
            else np.nan
        ),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Brier_Score": brier_score_loss(y_true, probability),
        "True_Negative": int(tn),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "True_Positive": int(tp),
    }


def bootstrap_auc_interval(y_true, probability, groups, iterations=2000):
    """Return a percentile 95% CI by resampling whole predictor-profile groups."""
    labels = np.asarray(y_true)
    scores = np.asarray(probability)
    group_ids = np.asarray(groups)
    unique_groups = np.unique(group_ids)
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indexes = np.concatenate(
            [np.flatnonzero(group_ids == group_id) for group_id in sampled_groups]
        )
        sample_labels = labels[indexes]
        if np.unique(sample_labels).size == 2:
            values.append(roc_auc_score(sample_labels, scores[indexes]))
    if not values:
        return [None, None]
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def save_subgroup_metrics(features, labels, probabilities):
    """Report test performance by gender and age band with group sizes."""
    rows = []
    groups = pd.DataFrame(index=features.index)
    if "gender" in features:
        groups["Gender"] = features["gender"].astype(str).str.title()
    if "age" in features:
        groups["Age_Group"] = pd.cut(
            pd.to_numeric(features["age"], errors="coerce"),
            bins=[-np.inf, 29, 44, 59, np.inf],
            labels=["16-29", "30-44", "45-59", "60+"],
        ).astype("string")

    label_series = pd.Series(np.asarray(labels), index=features.index)
    probability_series = pd.Series(np.asarray(probabilities), index=features.index)
    for group_name in groups:
        for group_value, index in groups.groupby(group_name, dropna=False).groups.items():
            actual = label_series.loc[index]
            score = probability_series.loc[index]
            predicted = (score >= DECISION_THRESHOLD).astype(int)
            row = {
                "Group_Type": group_name,
                "Group": str(group_value),
                "Patients": int(len(index)),
                "Actual_Positive": int(actual.sum()),
            }
            row.update(binary_metrics(actual, predicted, score))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "subgroup_metrics.csv", index=False)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_FILE)
    required_columns = {"patient_id", "target"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")
    if data["target"].isna().any() or data["target"].nunique() != 2:
        raise ValueError("The target must contain two classes and no missing values.")

    features = data.drop(columns=["patient_id", "target"])
    labels = data["target"].astype(int)
    categorical_columns = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = features.select_dtypes(exclude=["object", "string"]).columns.tolist()
    # Identical questionnaire profiles are treated as one group. This prevents
    # repeated profiles from leaking across the train/test or CV boundaries.
    group_ids = pd.util.hash_pandas_object(
        features.astype("string").fillna("<MISSING>"), index=False
    ).to_numpy()
    outer_split = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_indexes, test_indexes = next(
        outer_split.split(features, labels, groups=group_ids)
    )
    x_train = features.iloc[train_indexes]
    x_test = features.iloc[test_indexes]
    y_train = labels.iloc[train_indexes]
    y_test = labels.iloc[test_indexes]
    training_groups = group_ids[train_indexes]
    test_groups = group_ids[test_indexes]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_specs = {
        "Logistic_Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.1, 1, 10]},
        ),
        "Random_Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [2, 4],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            },
        ),
    }

    searches = {}
    comparison_rows = []
    for name, (estimator, parameter_grid) in model_specs.items():
        print(f"\nTuning {name} with stratified 5-fold cross-validation...")
        search = GridSearchCV(
            make_pipeline(estimator, numeric_columns, categorical_columns),
            parameter_grid,
            scoring="roc_auc",
            cv=folds,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(x_train, y_train, groups=training_groups)
        searches[name] = search
        comparison_rows.append(
            {
                "Model": name,
                "CV_ROC_AUC_Mean": float(search.best_score_),
                "CV_ROC_AUC_Std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "Best_Params": str(search.best_params_),
            }
        )
        joblib.dump(search.best_estimator_, MODEL_DIR / f"{name}.joblib")

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "CV_ROC_AUC_Mean", ascending=False
    )
    # Select the model using training-only cross-validation. The held-out test set
    # is used once for the final comparison and is not used for model selection.
    best_name = str(comparison.iloc[0]["Model"])
    best_model = searches[best_name].best_estimator_
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    metric_rows = []
    test_predictions = {}
    for name, search in searches.items():
        probability = search.best_estimator_.predict_proba(x_test)[:, 1]
        prediction = (probability >= DECISION_THRESHOLD).astype(int)
        metrics = binary_metrics(y_test, prediction, probability)
        cv_row = comparison.loc[comparison["Model"] == name].iloc[0]
        metric_rows.append(
            {
                "Model": name,
                "CV_ROC_AUC_Mean": cv_row["CV_ROC_AUC_Mean"],
                "CV_ROC_AUC_Std": cv_row["CV_ROC_AUC_Std"],
                **metrics,
                "Best_Params": str(search.best_params_),
            }
        )
        test_predictions[name] = probability

    metrics_table = pd.DataFrame(metric_rows).sort_values(
        "CV_ROC_AUC_Mean", ascending=False
    )
    metrics_table.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    best_probability = test_predictions[best_name]
    best_prediction = (best_probability >= DECISION_THRESHOLD).astype(int)
    test_output = x_test.copy()
    test_output.insert(0, "patient_id", data.loc[x_test.index, "patient_id"])
    test_output["actual_risk"] = y_test
    test_output["predicted_risk"] = best_prediction
    test_output["risk_probability"] = best_probability
    test_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    save_subgroup_metrics(x_test, y_test, best_probability)

    # ROC curves use the same held-out test rows for every candidate model.
    plt.figure(figsize=(8, 6))
    for name, probability in test_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        auc = roc_auc_score(y_test, probability)
        plt.plot(fpr, tpr, label=f"{name} (test AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="No-skill reference")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Held-Out Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve_comparison.png", dpi=160)
    plt.close()

    confusion = confusion_matrix(y_test, best_prediction, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    display = axis.imshow(confusion, cmap="Blues")
    fig.colorbar(display, ax=axis)
    axis.set_title(f"Held-Out Test Confusion Matrix: {best_name}")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_xticks([0, 1], labels=["Negative", "Positive"])
    axis.set_yticks([0, 1], labels=["Negative", "Positive"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_best.png", dpi=160)
    plt.close(fig)

    fraction_positive, mean_probability = calibration_curve(
        y_test, best_probability, n_bins=8, strategy="quantile"
    )
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(mean_probability, fraction_positive, marker="o", label=best_name)
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive fraction")
    axis.set_title("Probability Calibration on Held-Out Test Set")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_curve_best.png", dpi=160)
    plt.close(fig)

    # Feature and SHAP explanations are generated from the selected model.
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_classifier = best_model.named_steps["model"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    if hasattr(fitted_classifier, "coef_"):
        importance_values = np.abs(fitted_classifier.coef_[0])
    else:
        importance_values = fitted_classifier.feature_importances_
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance_values}
    ).sort_values("Importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    transformed_test = fitted_preprocessor.transform(x_test)
    if hasattr(transformed_test, "toarray"):
        transformed_test = transformed_test.toarray()
    try:
        if best_name in {"Random_Forest", "XGBoost"}:
            explainer = shap.TreeExplainer(fitted_classifier)
        else:
            background = fitted_preprocessor.transform(x_train)
            if hasattr(background, "toarray"):
                background = background.toarray()
            explainer = shap.LinearExplainer(fitted_classifier, background)
        shap_values = explainer.shap_values(transformed_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        elif np.ndim(shap_values) == 3:
            shap_values = shap_values[:, :, -1]
        mean_absolute_shap = np.abs(shap_values).mean(axis=0)
        shap_table = pd.DataFrame(
            {"Feature": feature_names, "Mean_Abs_SHAP": mean_absolute_shap}
        ).sort_values("Mean_Abs_SHAP", ascending=False)
        shap_table.to_csv(OUTPUT_DIR / "shap_global_importance.csv", index=False)
        shap_table.head(15).sort_values("Mean_Abs_SHAP").plot.barh(
            x="Feature", y="Mean_Abs_SHAP", legend=False, figsize=(9, 7)
        )
        plt.xlabel("Mean absolute SHAP value")
        plt.title(f"Global SHAP Importance: {best_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_global_bar.png", dpi=160)
        plt.close()
        (OUTPUT_DIR / "shap_error.txt").unlink(missing_ok=True)
    except Exception as error:  # SHAP support varies between estimator versions.
        (OUTPUT_DIR / "shap_error.txt").write_text(str(error), encoding="utf-8")
        print(f"SHAP explanation could not be generated: {error}")

    # Power BI table: actual cohort labels plus model scores for each row.
    all_probability = best_model.predict_proba(features)[:, 1]
    power_bi = data.copy()
    power_bi["risk_probability"] = all_probability
    power_bi["risk_category"] = pd.cut(
        all_probability,
        [-0.001, 0.33, 0.66, 1.001],
        labels=["Low", "Medium", "High"],
    ).astype("string")
    power_bi["prediction"] = np.where(
        all_probability >= DECISION_THRESHOLD, "Positive", "Negative"
    )
    power_bi["record_type"] = "Dataset observation"
    power_bi["evaluation_partition"] = "Training observation (in-sample score)"
    power_bi.loc[x_test.index, "evaluation_partition"] = "Held-out test observation"
    power_bi.to_csv(OUTPUT_DIR / "powerbi_patient_risk.csv", index=False)

    best_metrics = metrics_table.loc[metrics_table["Model"] == best_name].iloc[0]
    auc_interval = bootstrap_auc_interval(y_test, best_probability, test_groups)
    summary = {
        "best_model_selected_by": "mean stratified 5-fold CV ROC-AUC on training data",
        "best_model": best_name,
        "split_strategy": "StratifiedGroupKFold by identical predictor profile",
        "random_state": RANDOM_STATE,
        "target_test_fraction": TEST_SIZE,
        "actual_test_fraction": float(len(x_test) / len(data)),
        "decision_threshold": DECISION_THRESHOLD,
        "dataset_rows": int(len(data)),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "unique_predictor_profiles": int(np.unique(group_ids).size),
        "test_predictor_profiles": int(np.unique(test_groups).size),
        "positive_rate": float(labels.mean()),
        "test_roc_auc": float(best_metrics["ROC_AUC"]),
        "test_roc_auc_bootstrap_95pct_ci": auc_interval,
        "test_brier_score": float(best_metrics["Brier_Score"]),
        "python_packages": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nHeld-out test-set model comparison (models ranked by training CV ROC-AUC):")
    print(
        metrics_table[
            ["Model", "CV_ROC_AUC_Mean", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Brier_Score"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nSelected model: {best_name}")
    print(f"Test ROC-AUC 95% bootstrap interval: {auc_interval}")
    print(f"Subgroup analysis: {OUTPUT_DIR / 'subgroup_metrics.csv'}")


if __name__ == "__main__":
    main()
