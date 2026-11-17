
import pandas as pd
import numpy as np

from config import DATA_PATH, OUTPUTS_DIR

Q = 8800

print("Loading original hydraulic data...")

df = pd.read_parquet(
    DATA_PATH,
    columns=[
        "Q", "X", "Y",
        "h_avg", "rv_avg",
        "FHI", "report_FHI",
        "report_eligible"
    ]
)

df = df[
    (df["Q"] == Q) &
    (df["report_eligible"] == True)
].copy()

print("\nHydraulic data at Q =", Q)

print("\nOverall statistics:")
print(df[["h_avg", "rv_avg", "FHI"]].describe(
    percentiles=[0.95, 0.99, 0.999]
))

# Inspect extreme FHI cells
extreme = df[df["report_FHI"] >= 100].copy()

extreme = extreme.sort_values(
    "report_FHI",
    ascending=False
)

print("\nNumber of cells with FHI >= 100:", len(extreme))

print("\nTop 20 extreme FHI cells:")
print(
    extreme[
        ["X", "Y", "h_avg", "rv_avg", "FHI", "report_FHI"]
    ].head(20).to_string(index=False)
)

print("\nExtreme-cell depth statistics:")
print(extreme["h_avg"].describe())

print("\nExtreme-cell velocity statistics:")
print(extreme["rv_avg"].describe())

extreme.to_csv(
    OUTPUTS_DIR / "extreme_fhi_cells_q8800.csv",
    index=False
)

print("\nSaved extreme_fhi_cells_q8800.csv")
