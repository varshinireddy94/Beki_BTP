
import pandas as pd
import numpy as np

from config import OUTPUTS_DIR

path = OUTPUTS_DIR / "terrain_features_q200.csv"

df = pd.read_csv(path)
slope = df["slope_degrees"]

print("\nSLOPE PERCENTILES")
print(slope.quantile([
    0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0
]))

print("\nSLOPE RANGE COUNTS")
for threshold in [1, 2, 5, 10, 20, 30, 40]:
    count = int((slope > threshold).sum())
    print(
        f"Cells with slope > {threshold} degrees: "
        f"{count:,} ({count / len(df) * 100:.3f}%)"
    )

print("\nTOP 15 STEEPEST CELLS")
print(
    df.nlargest(15, "slope_degrees").to_string(index=False)
)

print("\nSLOPE VALIDATION COMPLETED")

