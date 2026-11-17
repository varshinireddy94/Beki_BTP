
import pandas as pd

from config import DATA_PATH

df = pd.read_parquet(
    DATA_PATH,
    filters=[("Q", "==", 200)],
    columns=[
        "row", "col", "X", "Y", "bed_elevation"
    ]
)

# Inspect a 5 x 5 neighbourhood
region = df[
    df["row"].between(207, 211) &
    df["col"].between(11, 15)
].sort_values(["row", "col"])

print("\nSTEEP REGION: ROWS 207-211, COLS 11-15")
print(region.to_string(index=False))

print("\nELEVATION GRID")
print(
    region.pivot(
        index="row",
        columns="col",
        values="bed_elevation"
    )
)

print("\nREGION INSPECTION COMPLETED")
