"""Build a concise Markdown experiment report from the latest run artifacts."""

import json
import html
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def markdown_to_html(markdown_text):
    """Render the generated report's headings, lists, tables, and paragraphs as HTML."""
    output = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Final Experiment Report</title>",
        "<style>body{font:16px/1.55 Arial,sans-serif;color:#1f2937;background:#f3f6fa;margin:0;padding:24px}main{max-width:1100px;margin:auto;background:#fff;padding:28px;border-radius:10px}h1,h2{color:#163b65}table{border-collapse:collapse;width:100%;margin:14px 0}th,td{text-align:left;padding:8px;border:1px solid #cbd5e1}th{background:#e2e8f0}li{margin:4px 0}.code{white-space:pre-wrap;background:#f8fafc;padding:12px;border-radius:6px}</style>",
        "</head><body><main>",
    ]
    in_list = False
    in_table = False
    in_code = False

    def close_blocks():
        nonlocal in_list, in_table
        if in_list:
            output.append("</ul>")
            in_list = False
        if in_table:
            output.append("</tbody></table>")
            in_table = False

    for line in markdown_text.splitlines():
        if line.strip().startswith("```"):
            close_blocks()
            output.append("</code></pre>" if in_code else "<pre class='code'><code>")
            in_code = not in_code
            continue
        if in_code:
            output.append(html.escape(line) + "\n")
            continue
        if not line.strip():
            close_blocks()
            continue
        if line.startswith("|"):
            if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line):
                continue
            if in_list:
                close_blocks()
            if not in_table:
                output.append("<table><tbody>")
                in_table = True
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            tag = "th" if output[-1] == "<table><tbody>" else "td"
            output.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            close_blocks()
            level = len(heading.group(1))
            output.append(f"<h{level}>{html.escape(heading.group(2))}</h{level}>")
        elif line.startswith("- "):
            if in_table:
                close_blocks()
            if not in_list:
                output.append("<ul>")
                in_list = True
            content = html.escape(line[2:])
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            output.append(f"<li>{content}</li>")
        else:
            close_blocks()
            content = html.escape(line)
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            output.append(f"<p>{content}</p>")
    close_blocks()
    if in_code:
        output.append("</code></pre>")
    output.append("</main></body></html>")
    return "\n".join(output)


def metric_table(metrics):
    columns = [
        "Model", "CV_ROC_AUC_Mean", "Accuracy", "Precision", "Recall", "F1",
        "ROC_AUC", "Specificity", "Brier_Score",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in metrics[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.3f}" if isinstance(value, (int, float)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    summary_file = OUTPUT_DIR / "run_summary.json"
    metrics_file = OUTPUT_DIR / "model_metrics.csv"
    quality_file = OUTPUT_DIR / "data_quality_report.txt"
    for required_file in (summary_file, metrics_file, quality_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing {required_file.name}. Run the full project pipeline first."
            )

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_file)
    quality = quality_file.read_text(encoding="utf-8").strip()
    interval = summary.get("test_roc_auc_bootstrap_95pct_ci", [None, None])
    subgroup_file = OUTPUT_DIR / "subgroup_metrics.csv"
    subgroup_table = "Subgroup analysis artifact was not generated."
    if subgroup_file.exists():
        subgroup = pd.read_csv(subgroup_file)
        subgroup_columns = [
            "Group_Type", "Group", "Patients", "Actual_Positive", "Recall",
            "Specificity", "ROC_AUC",
        ]
        subgroup_lines = [
            "| " + " | ".join(subgroup_columns) + " |",
            "|" + "|".join(["---"] * len(subgroup_columns)) + "|",
        ]
        for _, row in subgroup[subgroup_columns].iterrows():
            values = []
            for column in subgroup_columns:
                value = row[column]
                values.append(
                    f"{value:.3f}"
                    if column in {"Recall", "Specificity", "ROC_AUC"}
                    and pd.notna(value)
                    else str(value)
                )
            subgroup_lines.append("| " + " | ".join(values) + " |")
        subgroup_table = "\n".join(subgroup_lines)

    lines = [
        "# Experiment Results",
        "",
        f"Generated at (UTC): {summary.get('generated_at_utc', 'not recorded')}",
        "",
        "## Run configuration",
        "",
        f"- Selected model: **{summary['best_model']}**",
        f"- Selection rule: {summary['best_model_selected_by']}",
        f"- Split: {summary.get('split_strategy', 'not recorded')}",
        f"- Rows: {summary['dataset_rows']} total; {summary['training_rows']} training; {summary['test_rows']} held-out test",
        f"- Unique predictor profiles: {summary.get('unique_predictor_profiles', 'not recorded')}",
        f"- Held-out test profiles: {summary.get('test_predictor_profiles', 'not recorded')}",
        f"- Target positive rate: {summary['positive_rate']:.1%}",
        f"- Classification threshold: {summary['decision_threshold']:.2f}",
        "",
        "## Model comparison",
        "",
        "Models are ranked by training-only cross-validation ROC-AUC. Test metrics are reported on the held-out partition.",
        "",
        metric_table(metrics),
        "",
        f"Selected model test ROC-AUC 95% profile-bootstrap interval: {interval[0]} to {interval[1]}.",
        f"Selected model test Brier score: {summary['test_brier_score']:.3f}.",
        "",
        "## Data quality",
        "",
        "```text",
        quality,
        "```",
        "",
        "## Exploratory subgroup metrics",
        "",
        "These test-set subgroup results are descriptive. Include group sizes and avoid treating small groups as reliable fairness estimates.",
        "",
        subgroup_table,
        "",
        "## Interpretation and limitations",
        "",
        "This project analyzes a small public questionnaire dataset. Identical predictor profiles are grouped across splits, but the source has no person identifier, so profile groups cannot be confirmed as repeated or distinct people. The model score is not clinically calibrated or validated, and the model is not for diagnosis or patient-care decisions. External validation is needed before any real-world use.",
        "",
        "The model explanations describe fitted model behavior; they do not show causation. The bootstrap interval reflects held-out profile-sample uncertainty and excludes dataset shift and model-selection uncertainty.",
        "",
    ]
    report_path = OUTPUT_DIR / "experiment_report.md"
    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    html_path = OUTPUT_DIR / "experiment_report.html"
    html_path.write_text(markdown_to_html(report_content), encoding="utf-8")
    print(f"Experiment report saved: {report_path}")
    print(f"Browser version saved: {html_path}")


if __name__ == "__main__":
    main()
