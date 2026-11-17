
from pathlib import Path
import sys
import json
import gc

import numpy as np
import pandas as pd
import joblib

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR, MODELS_DIR

TRAIN_PATH = OUTPUTS_DIR / "train_fhi_with_slope.parquet"
VAL_PATH = OUTPUTS_DIR / "validation_fhi_with_slope.parquet"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)

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

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Features:", FEATURES)


# --------------------------------------------------
# 3. HYPERPARAMETER CONFIGURATIONS
# --------------------------------------------------

CONFIGS = [
    {
        "name": "Config_1",
        "num_leaves": 31,
        "min_child_samples": 100,
        "learning_rate": 0.03,
        "n_estimators": 4000
    },
    {
        "name": "Config_2",
        "num_leaves": 63,
        "min_child_samples": 100,
        "learning_rate": 0.03,
        "n_estimators": 4000
    },
    {
        "name": "Config_3",
        "num_leaves": 31,
        "min_child_samples": 200,
        "learning_rate": 0.03,
        "n_estimators": 4000
    },
    {
        "name": "Config_4",
        "num_leaves": 63,
        "min_child_samples": 200,
        "learning_rate": 0.05,
        "n_estimators": 3000
    }
]


# --------------------------------------------------
# 4. TRAIN AND EVALUATE
# --------------------------------------------------

results = []
best_mae = float("inf")
best_model = None
best_name = None

for config in CONFIGS:

    print("\n" + "=" * 60)
    print("Training:", config["name"])
    print(config)
    print("=" * 60)

    model = LGBMRegressor(
        objective="regression",
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
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
            early_stopping(150),
            log_evaluation(200)
        ]
    )

    # Predict in original FHI scale
    pred_log = model.predict(X_val)

    y_pred = np.maximum(
        0,
        np.expm1(pred_log)
    )

    mae = mean_absolute_error(y_val_actual, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_val_actual, y_pred)
    )
    r2 = r2_score(y_val_actual, y_pred)

    best_iteration = model.best_iteration_

    print("\nResults:", config["name"])
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Best iteration: {best_iteration}")

    results.append({
        "config": config["name"],
        "num_leaves": config["num_leaves"],
        "min_child_samples": config["min_child_samples"],
        "learning_rate": config["learning_rate"],
        "best_iteration": best_iteration,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # Select best model by validation MAE
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = config["name"]

    gc.collect()


# --------------------------------------------------
# 5. SAVE COMPARISON RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

results_path = OUTPUTS_DIR / "lightgbm_tuning_results.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("TUNING COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# --------------------------------------------------
# 6. SAVE BEST MODEL
# --------------------------------------------------

best_model_path = MODELS_DIR / "beki_fhi_lightgbm_tuned.pkl"

joblib.dump(best_model, best_model_path)

best_metrics = results_df.iloc[0].to_dict()

metrics_path = OUTPUTS_DIR / "best_tuned_lightgbm_metrics.json"

with open(metrics_path, "w") as f:
    json.dump(best_metrics, f, indent=4)

print("\nBEST MODEL:", best_name)
print("Best validation MAE:", best_mae)
print("Saved model:", best_model_path)
print("Saved comparison:", results_path)

print("\nTUNING COMPLETED SUCCESSFULLY!")
