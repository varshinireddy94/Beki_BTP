"""Shared constants for the BEKI FLOOD WATCH dashboard."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "beki_fhi_xgboost_tuned.pkl"
APP_DATA_DIR = PROJECT_DIR / "app_data"
GRID_PATH = APP_DATA_DIR / "beki_grid_static.parquet"
MASKS_PATH = APP_DATA_DIR / "simulated_eligibility_masks.npz"
BUILD_INFO_PATH = APP_DATA_DIR / "build_info.json"
METRICS_PATH = PROJECT_DIR / "outputs" / "xgboost_best_metrics.json"

# Exact feature order used in training (scripts/23_train_xgboost.py)
FEATURES = ["Q", "X", "Y", "bed_elevation", "NDWI_avg", "slope_degrees"]

N_ROWS, N_COLS = 286, 600
N_CELLS = N_ROWS * N_COLS

Q_MIN, Q_MAX = 200, 10_000
VALIDATION_Q = [800, 3200, 6200, 8800]
TEST_Q = [1200, 3600, 6800, 9200]

# Coordinate reference system.
# The source data stores projected X/Y in metres without a CRS tag. Under
# WGS 84 / UTM zone 46N the grid falls on the Beki River (90.78-91.05 E,
# 26.26-26.80 N), so that CRS is used for the basemap and GeoTIFF exports.
CRS_EPSG = 32646
CRS_NAME = "WGS 84 / UTM zone 46N (EPSG:32646)"
CRS_NOTE = (
    "X/Y are stored in metres without a CRS tag in the source data; "
    "WGS 84 / UTM zone 46N is inferred from the coordinate ranges, which place "
    "the grid on the Beki River. Change CRS_EPSG in app_config.py if your "
    "survey used a different projection."
)

MODEL_INFO = {
    "Algorithm": "XGBoost Regressor",
    "Configuration": "XGB_3",
    "Max depth": "10",
    "Learning rate": "0.03",
    "Min child weight": "10",
    "Estimators": "3,500",
    "Target": "log1p(report_FHI); predictions back-transformed with expm1",
    "Validation MAE": "0.8608",
    "Validation RMSE": "14.3140",
    "Validation R²": "0.5929",
}

# --------------------------------------------------------------------------
# Hazard classes (report thresholds - do not change)
# --------------------------------------------------------------------------
HAZARD_CLASSES = [
    # code, name, colour, rule
    (1, "Very Low", "#1b7837", "FHI ≤ 0.5"),
    (2, "Low", "#66bd63", "0.5 < FHI ≤ 1"),
    (3, "Moderate", "#fee50a", "1 < FHI ≤ 2"),
    (4, "High", "#f68d2e", "2 < FHI ≤ 3"),
    (5, "Very High", "#c51b1b", "FHI > 3"),
]
HAZARD_NAMES = {c: n for c, n, _, _ in HAZARD_CLASSES}
HAZARD_COLORS = {c: col for c, _, col, _ in HAZARD_CLASSES}
HAZARD_RULES = {c: r for c, _, _, r in HAZARD_CLASSES}

# Status codes for cells WITHOUT a valid prediction (never treated as safe)
NOPRED_OPEN_WATER = 6     # NDWI_avg >= 0: outside the model's training domain
NOPRED_OUTSIDE_SIM = 7    # outside the simulated h_avg > 0.1 m extent
NOPRED_INFO = {
    NOPRED_OPEN_WATER: (
        "No prediction – open water (NDWI ≥ 0)", "#8fa3b8",
        "Outside the model's training domain (NDWI ≥ 0). Hazard unknown.",
    ),
    NOPRED_OUTSIDE_SIM: (
        "No prediction – outside simulated extent", "#b9b9b9",
        "Not inundated above 0.1 m in the hydraulic simulation at this "
        "discharge, so the FHI model does not apply. Not assessed – do not "
        "treat as safe.",
    ),
}

COVERAGE_DOMAIN = "domain"
COVERAGE_SIMULATED = "simulated"
COVERAGE_LABELS = {
    COVERAGE_DOMAIN: "Model domain (all cells with NDWI < 0)",
    COVERAGE_SIMULATED: "Simulated inundation extent (h > 0.1 m)",
}
