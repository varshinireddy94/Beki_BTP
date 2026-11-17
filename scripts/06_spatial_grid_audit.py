
import pandas as pd
import numpy as np

from config import DATA_PATH

# Read one discharge scenario only
df = pd.read_parquet(
    DATA_PATH,
    filters=[("Q", "==", 200)],
    columns=[
        "Q", "row", "col", "X", "Y",
        "bed_elevation", "report_eligible"
    ]
)

print("Q=200 grid audit")
print("Total cells:", len(df))
print("Unique rows:", df["row"].nunique())
print("Unique columns:", df["col"].nunique())

print("\nCoordinate information")
print("Unique X coordinates:", df["X"].nunique())
print("Unique Y coordinates:", df["Y"].nunique())

print("\nCoordinate spacing")
x_spacing = np.sort(df["X"].unique())
y_spacing = np.sort(df["Y"].unique())

print("X spacing median:",
      np.median(np.diff(x_spacing)))
print("Y spacing median:",
      np.median(np.diff(y_spacing)))

print("\nBed elevation")
print("Minimum:", df["bed_elevation"].min())
print("Maximum:", df["bed_elevation"].max())
print("Mean:", df["bed_elevation"].mean())

print("\nSpatial audit completed!")
