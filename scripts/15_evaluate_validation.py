
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import OUTPUTS_DIR, MODELS_DIR

FEATURES = ["Q", "X", "Y", "bed_elevation", "NDWI_avg"]

print("Loading model and validation data...")

model = joblib.load(MODELS_DIR / "beki_fhi_lightgbm.pkl")
df = pd.read_parquet(OUTPUTS_DIR / "validation_fhi.parquet")

X = df[FEATURES].astype("float32")
actual = df["report_FHI"].to_numpy(dtype="float32")

pred_log = model.predict(X)
predicted = np.maximum(0, np.expm1(pred_log))

df["predicted_FHI"] = predicted
df["absolute_error"] = np.abs(actual - predicted)

print("\nPER-DISCHARGE VALIDATION RESULTS")
print("--------------------------------")

results = []

for q, group in df.groupby("Q"):
    y_true = group["report_FHI"].to_numpy()
    y_pred = group["predicted_FHI"].to_numpy()

    results.append({
        "Q": int(q),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Cells": len(group)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\nFEATURE IMPORTANCE")
print("------------------")

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))

results_df.to_csv(
    OUTPUTS_DIR / "validation_by_discharge.csv",
    index=False
)

print("\nSaved validation_by_discharge.csv")
