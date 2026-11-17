
import numpy as np
import pandas as pd
import joblib

from config import OUTPUTS_DIR, MODELS_DIR

FEATURES = ["Q", "X", "Y", "bed_elevation", "NDWI_avg"]

df = pd.read_parquet(OUTPUTS_DIR / "validation_fhi.parquet")

model = joblib.load(MODELS_DIR / "beki_fhi_lightgbm.pkl")

X = df[FEATURES].astype("float32")

df["predicted_FHI"] = np.maximum(
    0,
    np.expm1(model.predict(X))
)

df["actual_FHI"] = df["report_FHI"]

df["error"] = df["predicted_FHI"] - df["actual_FHI"]

# Focus on the difficult high-discharge scenario
high = df[df["Q"] == 8800].copy()

bins = [0, 1, 3, 10, 25, 50, 100, 500, np.inf]

high["FHI_range"] = pd.cut(
    high["actual_FHI"],
    bins=bins,
    right=False
)

summary = high.groupby(
    "FHI_range",
    observed=False
).agg(
    cells=("actual_FHI", "size"),
    mean_actual=("actual_FHI", "mean"),
    mean_predicted=("predicted_FHI", "mean"),
    mean_absolute_error=(
        "error",
        lambda x: np.abs(x).mean()
    ),
    mean_signed_error=("error", "mean")
).reset_index()

print("\nERROR ANALYSIS FOR Q = 8800")
print("--------------------------------")
print(summary.to_string(index=False))

print("\nOVERALL PREDICTION BIAS")
print("--------------------------------")
print("Mean actual:", high["actual_FHI"].mean())
print("Mean predicted:", high["predicted_FHI"].mean())
print("Mean signed error:", high["error"].mean())

summary.to_csv(
    OUTPUTS_DIR / "high_discharge_error_analysis.csv",
    index=False
)

print("\nSaved high_discharge_error_analysis.csv")
