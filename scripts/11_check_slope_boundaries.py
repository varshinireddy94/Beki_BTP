
import pandas as pd

from config import OUTPUTS_DIR

df = pd.read_csv(
    OUTPUTS_DIR / "terrain_features_q200.csv"
)

max_row = df["row"].max()
max_col = df["col"].max()

df["boundary"] = (
    (df["row"] == df["row"].min()) |
    (df["row"] == max_row) |
    (df["col"] == df["col"].min()) |
    (df["col"] == max_col)
)

print("\nSLOPE BY GRID LOCATION")
print(
    df.groupby("boundary")["slope_degrees"]
    .agg(["count", "mean", "median", "max"])
)

print("\nCELLS ABOVE 20 DEGREES")
print(
    df[df["slope_degrees"] > 20]
    .sort_values("slope_degrees", ascending=False)
    .head(20)
    .to_string(index=False)
)
