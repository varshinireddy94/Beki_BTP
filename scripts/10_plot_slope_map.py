
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_PATH, OUTPUTS_DIR

# Read terrain coordinates from one scenario
df = pd.read_parquet(
    DATA_PATH,
    filters=[("Q", "==", 200)],
    columns=["row", "col", "X", "Y"]
)

slope_df = pd.read_csv(
    OUTPUTS_DIR / "terrain_features_q200.csv"
)

df = df.merge(
    slope_df,
    on=["row", "col"],
    validate="one_to_one"
)

df = df.sort_values(["row", "col"])

rows = df["row"].nunique()
cols = df["col"].nunique()

X = df["X"].to_numpy().reshape(rows, cols)
Y = df["Y"].to_numpy().reshape(rows, cols)
S = df["slope_degrees"].to_numpy().reshape(rows, cols)

fig, ax = plt.subplots(figsize=(12, 8))

mesh = ax.pcolormesh(
    X, Y, S,
    shading="nearest",
    cmap="terrain",
    vmin=0,
    vmax=10
)

plt.colorbar(
    mesh,
    ax=ax,
    label="Terrain slope (degrees)"
)

ax.set_title("Beki River Basin - Terrain Slope Map")
ax.set_xlabel("UTM X coordinate (m)")
ax.set_ylabel("UTM Y coordinate (m)")

plt.tight_layout()

output = OUTPUTS_DIR / "terrain_slope_map.png"
plt.savefig(output, dpi=200)
plt.show()

print("Slope map saved:", output)
