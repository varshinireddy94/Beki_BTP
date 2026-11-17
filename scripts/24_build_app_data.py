"""
Build the static data package used by the Streamlit dashboard.

Outputs (in <project>/app_data/):
  beki_grid_static.parquet
      One row per grid cell (286 x 600 = 171,600), sorted by (row, col):
      row, col, X, Y, bed_elevation, NDWI_avg, slope_degrees, cell_area_m2
  simulated_eligibility_masks.npz
      For each of the 48 hydraulically simulated discharges, the boolean
      report-eligibility mask (h_avg > 0.1 AND NDWI_avg < 0), bit-packed.
      These are SIMULATION RESULTS, available only at those 48 discharges.
  build_info.json
      Provenance and audit numbers.

Static features (X, Y, bed_elevation, NDWI_avg) were verified to be identical
across discharge scenarios. Slope is taken from terrain_features_q200.csv,
exactly as merged into the training data (script 20).
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(r"E:\btp files\Beki_ML_Dataset_48Q.parquet")
SLOPE_PATH = PROJECT_DIR / "outputs" / "terrain_features_q200.csv"
OUT_DIR = PROJECT_DIR / "app_data"
OUT_DIR.mkdir(exist_ok=True)

N_ROWS, N_COLS = 286, 600
STATIC = ["X", "Y", "bed_elevation", "NDWI_avg"]


def cell_corner_nodes(A):
    """Corner nodes of a curvilinear grid from its cell centres.

    Interior corners are the mean of the four surrounding centres; the outer
    ring of centres is linearly extrapolated first so edge cells get corners.
    Returns an (n_rows + 1, n_cols + 1) array.
    """
    p = np.pad(A, 1, mode="edge").astype(float)
    p[0, :] = 2 * p[1, :] - p[2, :]
    p[-1, :] = 2 * p[-2, :] - p[-3, :]
    p[:, 0] = 2 * p[:, 1] - p[:, 2]
    p[:, -1] = 2 * p[:, -2] - p[:, -3]
    return 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])


def quad_areas(NX, NY):
    x1, y1 = NX[:-1, :-1], NY[:-1, :-1]
    x2, y2 = NX[:-1, 1:], NY[:-1, 1:]
    x3, y3 = NX[1:, 1:], NY[1:, 1:]
    x4, y4 = NX[1:, :-1], NY[1:, :-1]
    signed = 0.5 * (
        (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2)
        + (x3 * y4 - x4 * y3) + (x4 * y1 - x1 * y4)
    )
    return signed


# --------------------------------------------------
# 1. Static grid
# --------------------------------------------------
print("Reading static grid (Q=200)...")
base = pd.read_parquet(
    DATA_PATH, filters=[("Q", "==", 200)], columns=["row", "col"] + STATIC
).sort_values(["row", "col"]).reset_index(drop=True)

assert len(base) == N_ROWS * N_COLS
assert (base["row"].to_numpy() == np.repeat(np.arange(N_ROWS), N_COLS)).all()
assert (base["col"].to_numpy() == np.tile(np.arange(N_COLS), N_ROWS)).all()

slope = pd.read_csv(SLOPE_PATH)
grid = base.merge(slope, on=["row", "col"], how="left", validate="one_to_one")
assert grid["slope_degrees"].notna().all()

X = grid["X"].to_numpy().reshape(N_ROWS, N_COLS)
Y = grid["Y"].to_numpy().reshape(N_ROWS, N_COLS)
signed = quad_areas(cell_corner_nodes(X), cell_corner_nodes(Y))
if not (signed > 0).all() and not (signed < 0).all():
    raise RuntimeError("Grid cells have inconsistent orientation (folded cells).")
grid["cell_area_m2"] = np.abs(signed).ravel()

# --------------------------------------------------
# 2. Simulated eligibility masks + static check
# --------------------------------------------------
print("Reading per-scenario eligibility masks...")
pf = pq.ParquetFile(DATA_PATH)
all_q = sorted(set(pq.read_table(DATA_PATH, columns=["Q"]).column("Q").to_pylist()))
masks = {}
for q in all_q:
    d = pd.read_parquet(
        DATA_PATH, filters=[("Q", "==", q)],
        columns=["row", "col", "report_eligible"] + STATIC,
    ).sort_values(["row", "col"]).reset_index(drop=True)
    for c in STATIC:
        if not np.allclose(d[c].to_numpy(), base[c].to_numpy()):
            raise RuntimeError(f"Static feature {c} differs at Q={q}")
    masks[q] = d["report_eligible"].to_numpy(dtype=bool)
    print(f"  Q={q:>5}: eligible cells = {masks[q].sum():,}")

q_arr = np.array(all_q, dtype=np.int32)
packed = np.stack([np.packbits(masks[q]) for q in all_q])
np.savez_compressed(
    OUT_DIR / "simulated_eligibility_masks.npz",
    discharges=q_arr, packed_masks=packed, n_cells=np.int64(N_ROWS * N_COLS),
)

grid.to_parquet(OUT_DIR / "beki_grid_static.parquet", index=False)

info = {
    "source_dataset": str(DATA_PATH),
    "slope_source": str(SLOPE_PATH),
    "n_rows": N_ROWS,
    "n_cols": N_COLS,
    "n_cells": int(len(grid)),
    "static_features_identical_across_Q": True,
    "simulated_discharges": [int(q) for q in all_q],
    "cell_area_method": (
        "Shoelace area of quadrilateral cells whose corner nodes are the mean "
        "of the four surrounding cell centres (edge ring linearly extrapolated)."
    ),
    "cell_area_m2_min": float(grid["cell_area_m2"].min()),
    "cell_area_m2_median": float(grid["cell_area_m2"].median()),
    "cell_area_m2_max": float(grid["cell_area_m2"].max()),
    "total_grid_area_km2": float(grid["cell_area_m2"].sum() / 1e6),
    "cells_ndwi_ge_0": int((grid["NDWI_avg"] >= 0).sum()),
}
(OUT_DIR / "build_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
print("\nApp data written to", OUT_DIR)
