
import pandas as pd
import numpy as np

from config import DATA_PATH, OUTPUTS_DIR

# Read one scenario: static terrain is identical across scenarios
df = pd.read_parquet(
    DATA_PATH,
    filters=[("Q", "==", 200)],
    columns=["row", "col", "X", "Y", "bed_elevation"]
)

df = df.sort_values(["row", "col"])

rows = df["row"].nunique()
cols = df["col"].nunique()

X = df["X"].to_numpy().reshape(rows, cols)
Y = df["Y"].to_numpy().reshape(rows, cols)
Z = df["bed_elevation"].to_numpy().reshape(rows, cols)

# Calculate coordinate derivatives
dZ_drow, dZ_dcol = np.gradient(Z)
dX_drow, dX_dcol = np.gradient(X)
dY_drow, dY_dcol = np.gradient(Y)

# Local coordinate transformation
det = (
    dX_dcol * dY_drow
    - dX_drow * dY_dcol
)

valid = np.abs(det) > 1e-10

dZ_dX = np.full_like(Z, np.nan)
dZ_dY = np.full_like(Z, np.nan)

dZ_dX[valid] = (
    dZ_dcol[valid] * dY_drow[valid]
    - dZ_drow[valid] * dY_dcol[valid]
) / det[valid]

dZ_dY[valid] = (
    dX_dcol[valid] * dZ_drow[valid]
    - dX_drow[valid] * dZ_dcol[valid]
) / det[valid]

slope = np.degrees(
    np.arctan(np.sqrt(dZ_dX**2 + dZ_dY**2))
)

# Save terrain features
terrain = df[["row", "col"]].copy()
terrain["slope_degrees"] = slope.ravel()

output = OUTPUTS_DIR / "terrain_features_q200.csv"
terrain.to_csv(output, index=False)

print("\nTERRAIN FEATURE SUMMARY")
print(terrain["slope_degrees"].describe())
print("\nInvalid slope cells:", terrain["slope_degrees"].isna().sum())
print("\nSaved:", output)
