
import pandas as pd
import numpy as np

from config import DATA_PATH

df = pd.read_parquet(
    DATA_PATH,
    filters=[("Q", "==", 200)],
    columns=["row", "col", "X", "Y", "bed_elevation"]
)

# Arrange cells according to their actual grid positions
df = df.sort_values(["row", "col"])

rows = df["row"].nunique()
cols = df["col"].nunique()

X = df["X"].to_numpy().reshape(rows, cols)
Y = df["Y"].to_numpy().reshape(rows, cols)
Z = df["bed_elevation"].to_numpy().reshape(rows, cols)

# Distance between horizontally adjacent cells
dx = np.sqrt(
    np.diff(X, axis=1)**2 +
    np.diff(Y, axis=1)**2
)

# Distance between vertically adjacent cells
dy = np.sqrt(
    np.diff(X, axis=0)**2 +
    np.diff(Y, axis=0)**2
)

print("\nHORIZONTAL NEIGHBOUR DISTANCES")
print("Minimum:", np.min(dx))
print("Median:", np.median(dx))
print("Maximum:", np.max(dx))

print("\nVERTICAL NEIGHBOUR DISTANCES")
print("Minimum:", np.min(dy))
print("Median:", np.median(dy))
print("Maximum:", np.max(dy))

print("\nGRID COORDINATE CHECK")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Elevation shape:", Z.shape)

print("\nGrid spacing audit completed!")
