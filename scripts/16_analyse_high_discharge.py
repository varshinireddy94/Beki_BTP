
import pandas as pd
import numpy as np

from config import OUTPUTS_DIR

df = pd.read_parquet(
    OUTPUTS_DIR / "validation_fhi.parquet"
)

print("FHI DISTRIBUTION BY VALIDATION DISCHARGE")
print("----------------------------------------")

for q in sorted(df["Q"].unique()):
    values = df.loc[df["Q"] == q, "report_FHI"]

    print(f"\nQ = {q}")
    print("Count:", len(values))
    print("Mean:", values.mean())
    print("Median:", values.median())
    print("Std:", values.std())
    print("95th percentile:", values.quantile(0.95))
    print("99th percentile:", values.quantile(0.99))
    print("99.9th percentile:", values.quantile(0.999))
    print("Maximum:", values.max())

print("\nEXTREME FHI COUNTS")
print("------------------")

for q in sorted(df["Q"].unique()):
    values = df.loc[df["Q"] == q, "report_FHI"]

    print(
        f"Q={q}: "
        f"FHI > 10: {(values > 10).sum():,}, "
        f"FHI > 50: {(values > 50).sum():,}, "
        f"FHI > 100: {(values > 100).sum():,}"
    )
