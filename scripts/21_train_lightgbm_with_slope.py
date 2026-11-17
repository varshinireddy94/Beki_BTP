
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR, MODELS_DIR

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

train_path = OUTPUTS_DIR / "train_fhi_with_slope.parquet"
val_path = OUTPUTS_DIR / "validation_fhi_with_slope.parquet"

print("Loading training and validation datasets...")

train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)

print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)

# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

FEATURES = [
    "Q",
    "X",
    "Y",
    "bed_elevation",
    "NDWI_avg",
    "slope_degrees"
]

TARGET = "report_FHI"

X_train = train_df[FEATURES]
X_val = val_df[FEATURES]

y_train = np.log1p(train_df[TARGET].values)
y_val = np.log1p(val_df[TARGET].values)

print("\nFeatures:", FEATURES)
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

# --------------------------------------------------
# 3. TRAIN LIGHTGBM
# --------------------------------------------------

print("\nTraining LightGBM with slope feature...")

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l2",
    callbacks=[
        early_stopping(100),
        log_evaluation(100)
    ]
)

# --------------------------------------------------
# 4. PREDICTIONS
# --------------------------------------------------

print("\nEvaluating model...")

pred_log = model.predict(X_val)

y_actual = np.expm1(y_val)
y_pred = np.maximum(0, np.expm1(pred_log))

# --------------------------------------------------
# 5. EVALUATION METRICS
# --------------------------------------------------

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("\n" + "=" * 50)
print("LIGHTGBM WITH SLOPE - VALIDATION RESULTS")
print("=" * 50)

print(f"MAE:            {mae:.6f}")
print(f"RMSE:           {rmse:.6f}")
print(f"R2 Score:       {r2:.6f}")
print(f"Best iteration: {model.best_iteration_}")

# --------------------------------------------------
# 6. FEATURE IMPORTANCE
# --------------------------------------------------

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    OUTPUTS_DIR / "feature_importance_with_slope.csv",
    index=False
)

# --------------------------------------------------
# 7. SAVE MODEL AND METRICS
# --------------------------------------------------

model_path = MODELS_DIR / "beki_fhi_lightgbm_with_slope.pkl"

joblib.dump(model, model_path)

metrics = {
    "model": "LightGBM with slope",
    "features": FEATURES,
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "best_iteration": int(model.best_iteration_)
}

metrics_path = OUTPUTS_DIR / "metrics_with_slope.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved:", model_path)
print("Metrics saved:", metrics_path)

print("\nTraining completed successfully!")
