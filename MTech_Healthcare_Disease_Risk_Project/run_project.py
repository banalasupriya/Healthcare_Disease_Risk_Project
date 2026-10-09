import os
import subprocess
import sys
from getpass import getpass
from pathlib import Path
import webbrowser
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
POWERBI_REPORT = ROOT / "powerbi file.pbix"
POWERBI_DATA = ROOT / "outputs" / "powerbi_patient_risk.csv"
PATIENT_RESULT = ROOT / "outputs" / "prediction_result.csv"
FINAL_REPORT = ROOT / "outputs" / "experiment_report.html"
load_dotenv(ROOT / ".env")

def run(script):
    print("\n" + "=" * 70, flush=True)
    print(f"RUNNING {script}", flush=True)
    print("=" * 70, flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True)

def open_power_bi_report():
    if not POWERBI_REPORT.exists():
        print(f"\nPower BI report not found: {POWERBI_REPORT}")
        return
    if os.name != "nt":
        print(f"\nOpen this Power BI report manually: {POWERBI_REPORT}")
        return
    try:
        os.startfile(str(POWERBI_REPORT))
        print(f"\nOpened Power BI report: {POWERBI_REPORT}")
    except OSError as error:
        print(f"\nCould not open the Power BI report: {error}")
        print(f"Open it manually: {POWERBI_REPORT}")

def open_final_report():
    if not FINAL_REPORT.exists():
        print(f"\nFinal report not found: {FINAL_REPORT}")
        return
    try:
        webbrowser.open(FINAL_REPORT.as_uri(), new=2)
        print(f"\nOpened final report: {FINAL_REPORT}")
    except OSError as error:
        print(f"\nCould not open the final report: {error}")
        print(f"Open it manually: {FINAL_REPORT}")

try:
    if not os.environ.get("SMTP_PASSWORD"):
        os.environ["SMTP_PASSWORD"] = getpass("Gmail app password (input hidden): ").strip()
    run("download_data.py")
    run("prepare_data.py")
    run("sql_analysis.py")
    run("train_models.py")
    print("\nTraining and analytics complete. Opening Power BI before the Diabetes Health Check.")
    open_power_bi_report()
    if POWERBI_REPORT.exists() and os.name == "nt":
        input("Review the Power BI dashboard. Press Enter here when you are ready to open the Diabetes Health Check in your browser...")
    print("\nThe Diabetes Health Check will open in your browser. Submit it to view the patient report; press Ctrl+C in this terminal when finished to generate and open the final experiment report.")
    run("predict_patient.py")
    run("generate_report.py")
    print("\nPROJECT COMPLETE")
    print(f"Power BI dataset: {POWERBI_DATA}")
    print(f"Single-patient result: {PATIENT_RESULT}")
    print("Refresh Power BI after the assessment to load the latest dashboard_data.csv values.")
    open_final_report()
except subprocess.CalledProcessError as e:
    print(f"\nERROR: {e}. Read the message above and fix the dependency or network issue.")
    raise
