
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import OUTPUTS_DIR, MODELS_DIR

FEATURES = ["Q", "X", "Y", "bed_elevation", "NDWI_avg"]
TARGET = "report_FHI"

print("Loading training data...")
train_df = pd.read_parquet(OUTPUTS_DIR / "train_fhi.parquet")

print("Loading validation data...")
val_df = pd.read_parquet(OUTPUTS_DIR / "validation_fhi.parquet")

X_train = train_df[FEATURES].astype("float32")
y_train = np.log1p(train_df[TARGET].astype("float32"))

X_val = val_df[FEATURES].astype("float32")
y_val_original = val_df[TARGET].to_numpy(dtype="float32")
y_val = np.log1p(y_val_original)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Features:", FEATURES)

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

print("\nStarting LightGBM training...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l2",
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
)

print("\nGenerating validation predictions...")
pred_log = model.predict(
    X_val,
    num_iteration=model.best_iteration_
)

pred = np.maximum(0, np.expm1(pred_log))

mae = mean_absolute_error(y_val_original, pred)
rmse = np.sqrt(mean_squared_error(y_val_original, pred))
r2 = r2_score(y_val_original, pred)

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "best_iteration": int(model.best_iteration_)
}

print("\nVALIDATION RESULTS")
print("------------------")
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    MODELS_DIR / "beki_fhi_lightgbm.pkl"
)

with open(OUTPUTS_DIR / "validation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
    OUTPUTS_DIR / "feature_importance.csv",
    index=False
)

print("\nModel and evaluation files saved successfully!")
