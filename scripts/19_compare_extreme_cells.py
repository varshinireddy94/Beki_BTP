
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from config import DATA_PATH, OUTPUTS_DIR

print("Loading hydraulic dataset...")

df = pd.read_parquet(
    DATA_PATH,
    columns=["Q", "cell_id", "h_avg", "rv_avg", "FHI"]
)

# Find the 20 highest FHI cells at Q = 8800
reference = df[df["Q"] == 8800].nlargest(20, "FHI")

cell_ids = reference["cell_id"].tolist()

# Compare the same cells across all available discharges
comparison = df[df["cell_id"].isin(cell_ids)].copy()

pivot = comparison.pivot(
    index="cell_id",
    columns="Q",
    values="FHI"
)

print("\nTOP 20 EXTREME CELLS AT Q = 8800")
print("--------------------------------")

for _, row in reference.iterrows():
    cell = row["cell_id"]

    print(f"\nCell ID: {cell}")
    print(f"FHI at 8800: {row['FHI']:.3f}")

    values = pivot.loc[cell].dropna()

    print(values.to_string())

pivot.to_csv(
    OUTPUTS_DIR / "extreme_cells_across_discharges.csv"
)

print("\nSaved comparison successfully!")
