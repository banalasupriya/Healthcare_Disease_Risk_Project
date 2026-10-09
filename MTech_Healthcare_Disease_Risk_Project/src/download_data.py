from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT = DATA_DIR / "diabetes_data_upload.csv"

if OUT.exists() and OUT.stat().st_size > 1000:
    print(f"Dataset already exists: {OUT}")
    raise SystemExit

print("Downloading UCI Early Stage Diabetes Risk Prediction dataset...")
ds = fetch_ucirepo(id=529)
X = ds.data.features.copy()
y = ds.data.targets.copy()
# UCI target column is the only target column.
target = y.columns[0]
df = X.copy()
df[target] = y[target]
df.to_csv(OUT, index=False)
print(f"Saved {len(df)} rows to {OUT}")
