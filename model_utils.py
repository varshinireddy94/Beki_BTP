"""Model loading, static-grid loading and FHI prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from app_config import (
    BUILD_INFO_PATH, COVERAGE_DOMAIN, COVERAGE_SIMULATED, FEATURES, GRID_PATH,
    MASKS_PATH, MODEL_PATH, N_CELLS, N_COLS, N_ROWS, NOPRED_OPEN_WATER,
    NOPRED_OUTSIDE_SIM, Q_MAX, Q_MIN, TEST_Q, VALIDATION_Q,
)


class AppDataError(RuntimeError):
    """Raised when the model or the static spatial data cannot be used."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_model():
    if not MODEL_PATH.exists():
        raise AppDataError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    names = list(getattr(model, "feature_names_in_", []))
    if names and names != FEATURES:
        raise AppDataError(
            f"Model expects features {names}, dashboard provides {FEATURES}."
        )
    return model


def load_grid() -> pd.DataFrame:
    if not GRID_PATH.exists():
        raise AppDataError(
            f"Static grid not found: {GRID_PATH}. "
            "Run scripts/24_build_app_data.py first."
        )
    grid = pd.read_parquet(GRID_PATH)
    need = ["row", "col", "X", "Y", "bed_elevation", "NDWI_avg",
            "slope_degrees", "cell_area_m2"]
    missing = [c for c in need if c not in grid.columns]
    if missing:
        raise AppDataError(f"Static grid is missing columns: {missing}")
    if len(grid) != N_CELLS:
        raise AppDataError(f"Expected {N_CELLS:,} cells, found {len(grid):,}.")
    grid = grid.sort_values(["row", "col"]).reset_index(drop=True)
    if grid[need].isna().any().any():
        raise AppDataError("Static grid contains missing values.")
    return grid


def load_simulated_masks() -> dict[int, np.ndarray]:
    """Report-eligibility masks from the 48 hydraulic simulations."""
    if not MASKS_PATH.exists():
        return {}
    z = np.load(MASKS_PATH)
    n = int(z["n_cells"])
    return {
        int(q): np.unpackbits(z["packed_masks"][i])[:n].astype(bool)
        for i, q in enumerate(z["discharges"])
    }


def load_build_info() -> dict:
    try:
        return json.loads(BUILD_INFO_PATH.read_text())
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------
def validate_discharge(q) -> float:
    try:
        q = float(q)
    except (TypeError, ValueError):
        raise ValueError("Discharge must be a number.")
    if not np.isfinite(q):
        raise ValueError("Discharge must be a finite number.")
    if q < Q_MIN or q > Q_MAX:
        raise ValueError(
            f"Discharge must be between {Q_MIN:,} and {Q_MAX:,} cumec "
            "(the range covered by the training simulations)."
        )
    return q


def scenario_status(q: float, simulated_q: list[int]) -> tuple[str, str]:
    """Return (short badge, explanation) describing how q relates to the data."""
    qi = int(round(q))
    if abs(q - qi) < 1e-9 and qi in simulated_q:
        if qi in VALIDATION_Q:
            return ("Simulated – validation scenario",
                    "This discharge was simulated hydraulically and held out "
                    "for model validation (not used for training).")
        if qi in TEST_Q:
            return ("Simulated – test scenario",
                    "This discharge was simulated hydraulically and held out "
                    "as an independent test (not used for training).")
        return ("Simulated – training scenario",
                "This discharge is one of the 40 hydraulic scenarios the "
                "model was trained on.")
    lower = max([s for s in simulated_q if s < q], default=None)
    upper = min([s for s in simulated_q if s > q], default=None)
    return ("Unsimulated discharge – ML interpolation",
            f"No hydraulic simulation exists for this discharge. The model "
            f"interpolates between the nearest simulated scenarios "
            f"({lower:,} and {upper:,} cumec).")


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
HAZARD_BINS = np.array([0.5, 1.0, 2.0, 3.0])


def classify_fhi(fhi: np.ndarray) -> np.ndarray:
    """Report thresholds: <=0.5 ->1, <=1 ->2, <=2 ->3, <=3 ->4, >3 ->5."""
    return (np.searchsorted(HAZARD_BINS, fhi, side="left") + 1).astype(np.int8)


def predict_domain_fhi(model, grid: pd.DataFrame, q: float) -> np.ndarray:
    """Predict FHI for every cell inside the model domain (NDWI_avg < 0).

    Returns a float32 array of length N_CELLS with NaN outside the domain.
    """
    domain = grid["NDWI_avg"].to_numpy() < 0
    X = grid.loc[domain, FEATURES[1:]].copy()
    X.insert(0, "Q", np.float64(q))
    X = X[FEATURES]
    pred_log = model.predict(X)
    fhi = np.maximum(0.0, np.expm1(pred_log))
    out = np.full(len(grid), np.nan, dtype=np.float32)
    out[domain] = fhi.astype(np.float32)
    return out


@dataclass
class Prediction:
    q: float
    coverage: str                 # COVERAGE_DOMAIN or COVERAGE_SIMULATED
    fhi: np.ndarray               # float32, NaN where no valid prediction
    status: np.ndarray            # int8: 1-5 hazard class, 6/7 no-prediction
    coverage_note: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return self.status <= 5

    def grid2d(self, arr: np.ndarray) -> np.ndarray:
        return arr.reshape(N_ROWS, N_COLS)


def make_prediction(domain_fhi: np.ndarray, grid: pd.DataFrame, q: float,
                    coverage: str, masks: dict[int, np.ndarray]) -> Prediction:
    """Apply the selected coverage rule to domain predictions."""
    ndwi_ok = grid["NDWI_avg"].to_numpy() < 0
    status = np.full(len(grid), NOPRED_OPEN_WATER, dtype=np.int8)

    qi = int(round(q))
    if coverage == COVERAGE_SIMULATED:
        if not (abs(q - qi) < 1e-9 and qi in masks):
            raise ValueError(
                "The simulated inundation extent is only available at the 48 "
                "hydraulically simulated discharges."
            )
        valid = masks[qi] & ndwi_ok
        status[ndwi_ok & ~valid] = NOPRED_OUTSIDE_SIM
        note = (f"Predictions shown only where the hydraulic simulation at "
                f"Q = {qi:,} cumec gives depth > 0.1 m and NDWI < 0 "
                f"(the model's training condition).")
    else:
        valid = ndwi_ok
        note = ("Predictions shown for every cell with NDWI < 0. The model "
                "estimates the hazard a cell would have IF it is inundated "
                "(> 0.1 m); it does not predict whether the cell floods.")

    fhi = np.where(valid, domain_fhi, np.nan).astype(np.float32)
    status[valid] = classify_fhi(fhi[valid])
    return Prediction(q=q, coverage=coverage, fhi=fhi, status=status,
                      coverage_note=note)
