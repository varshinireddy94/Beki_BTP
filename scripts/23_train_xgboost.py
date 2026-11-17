
from pathlib import Path
import sys
import gc
import json

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR, MODELS_DIR

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_parquet(
    OUTPUTS_DIR / "train_fhi_with_slope.parquet"
)

val_df = pd.read_parquet(
    OUTPUTS_DIR / "validation_fhi_with_slope.parquet"
)

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

y_train_actual = train_df[TARGET].to_numpy()
y_val_actual = val_df[TARGET].to_numpy()

y_train = np.log1p(y_train_actual)
y_val = np.log1p(y_val_actual)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# --------------------------------------------------
# 2. CONFIGURATIONS
# --------------------------------------------------

CONFIGS = [
    {
        "name": "XGB_1",
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "n_estimators": 3500
    },
    {
        "name": "XGB_2",
        "max_depth": 8,
        "learning_rate": 0.02,
        "min_child_weight": 10,
        "n_estimators": 4000
    },
    {
        "name": "XGB_3",
        "max_depth": 10,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "n_estimators": 3500
    },
    {
        "name": "XGB_4",
        "max_depth": 8,
        "learning_rate": 0.03,
        "min_child_weight": 20,
        "n_estimators": 3500
    }
]

results = []
best_model = None
best_mae = float("inf")
best_name = None

# --------------------------------------------------
# 3. TRAIN CONFIGURATIONS
# --------------------------------------------------

for config in CONFIGS:

    print("\n" + "=" * 60)
    print("Training:", config["name"])
    print(config)
    print("=" * 60)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=150
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=250
    )

    pred_log = model.predict(X_val)

    predictions = np.maximum(
        0,
        np.expm1(pred_log)
    )

    mae = mean_absolute_error(
        y_val_actual, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_val_actual, predictions)
    )

    r2 = r2_score(y_val_actual, predictions)

    print("\nResults:")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.6f}")
    print("Best iteration:", model.best_iteration)

    results.append({
        "config": config["name"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "min_child_weight": config["min_child_weight"],
        "best_iteration": model.best_iteration,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # Select by validation MAE for now
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = config["name"]

    gc.collect()

# --------------------------------------------------
# 4. COMPARE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\n" + "=" * 60)
print("XGBOOST TUNING RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUTS_DIR / "xgboost_tuning_results.csv",
    index=False
)

# --------------------------------------------------
# 5. SAVE BEST MODEL
# --------------------------------------------------

model_path = MODELS_DIR / "beki_fhi_xgboost_tuned.pkl"

joblib.dump(best_model, model_path)

best_metrics = results_df.iloc[0].to_dict()

with open(
    OUTPUTS_DIR / "xgboost_best_metrics.json",
    "w"
) as f:
    json.dump(best_metrics, f, indent=4)

print("\nBest configuration:", best_name)
print("Best MAE:", best_mae)
print("Model saved:", model_path)
print("\nXGBOOST TUNING COMPLETED!")
