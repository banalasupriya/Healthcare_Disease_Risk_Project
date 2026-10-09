"""Normalize the UCI questionnaire data and write a data quality summary."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "diabetes_data_upload.csv"
CLEAN_FILE = ROOT / "data" / "cleaned_diabetes.csv"
OUTPUT_DIR = ROOT / "outputs"
TARGET_CANDIDATES = ["class", "target", "outcome"]
TARGET_MAPPING = {
    "positive": 1,
    "negative": 0,
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
}


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_FILE}. Run src/download_data.py first."
        )

    data = pd.read_csv(RAW_FILE)
    data.columns = [
        str(column).strip().lower().replace(" ", "_") for column in data.columns
    ]
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Column names are duplicated after normalization: {duplicates}")

    for column in data.select_dtypes(include=["object", "string"]).columns:
        values = data[column].astype("string").str.strip().str.lower()
        data[column] = values.replace({"": pd.NA, "nan": pd.NA, "null": pd.NA})

    target_column = next(
        (column for column in TARGET_CANDIDATES if column in data.columns), None
    )
    if target_column is None:
        raise ValueError(
            "Could not identify target column. Expected one of "
            f"{TARGET_CANDIDATES}; found {data.columns.tolist()}"
        )

    mapped_target = data[target_column].astype("string").str.strip().str.lower().map(
        TARGET_MAPPING
    )
    if mapped_target.isna().any():
        bad_values = data.loc[mapped_target.isna(), target_column].unique().tolist()
        raise ValueError(f"Missing or unrecognized target values: {bad_values}")

    data["target"] = mapped_target.astype(int)
    if target_column != "target":
        data = data.drop(columns=[target_column])
    if "age" in data:
        data["age"] = pd.to_numeric(data["age"], errors="coerce")
    if "patient_id" in data.columns:
        data = data.drop(columns=["patient_id"])

    data.insert(
        0,
        "patient_id",
        [f"P{index:04d}" for index in range(1, len(data) + 1)],
    )

    feature_columns = [column for column in data.columns if column not in {"patient_id", "target"}]
    predictor_data = data[feature_columns]
    duplicate_predictor_rows = int(predictor_data.duplicated().sum())
    conflicting_duplicate_groups = int(
        data.groupby(feature_columns, dropna=False)["target"].nunique().gt(1).sum()
    )
    missing_values = data.drop(columns="patient_id").isna().sum()
    target_counts = data["target"].value_counts().sort_index()
    age_summary = "Age column not present."
    if "age" in data:
        ages = pd.to_numeric(data["age"], errors="coerce")
        age_summary = (
            f"Age range: {ages.min():g} to {ages.max():g}; "
            f"missing/non-numeric ages: {int(ages.isna().sum())}."
        )

    CLEAN_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    data.to_csv(CLEAN_FILE, index=False)

    report_lines = [
        "DATA QUALITY SUMMARY",
        "====================",
        f"Source file: {RAW_FILE.name}",
        f"Rows: {len(data)}",
        f"Predictor columns: {len(feature_columns)}",
        f"Missing predictor/target cells: {int(missing_values.sum())}",
        f"Rows with at least one missing value: {int(data.drop(columns='patient_id').isna().any(axis=1).sum())}",
        f"Duplicate predictor rows: {duplicate_predictor_rows}",
        f"Duplicate predictor groups with conflicting targets: {conflicting_duplicate_groups}",
        age_summary,
        "Target distribution:",
    ]
    for target_value, count in target_counts.items():
        report_lines.append(
            f"  {('Positive' if target_value else 'Negative')}: {int(count)} "
            f"({count / len(data):.1%})"
        )
    report_lines.extend(
        [
            "",
            "Duplicate patient records can make random train/test metrics optimistic;",
            "interpret the scores with this limitation in mind.",
        ]
    )
    report_file = OUTPUT_DIR / "data_quality_report.txt"
    report_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nDATA PREPARATION COMPLETE")
    print(f"Rows: {len(data)} | Predictor columns: {len(feature_columns)}")
    print(f"Clean data: {CLEAN_FILE}")
    print(f"Data quality report: {report_file}")
    class_counts = data["target"].value_counts().rename(index={0: "Negative", 1: "Positive"})
    print(
        "Class distribution: "
        f"Negative={int(class_counts.get('Negative', 0))}, "
        f"Positive={int(class_counts.get('Positive', 0))}"
    )


if __name__ == "__main__":
    main()
