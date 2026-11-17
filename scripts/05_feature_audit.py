
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import DATA_PATH

features = [
    "Q",
    "X",
    "Y",
    "bed_elevation",
    "NDWI_avg"
]

pf = pq.ParquetFile(DATA_PATH)

stats = {
    col: {
        "missing": 0,
        "min": np.inf,
        "max": -np.inf,
        "sum": 0.0,
        "count": 0
    }
    for col in features
}

for batch in pf.iter_batches(
    batch_size=200_000,
    columns=features + ["report_eligible"]
):
    df = batch.to_pandas()

    # Only inspect cells used for hazard-model training
    df = df[df["report_eligible"]]

    for col in features:
        values = df[col].to_numpy(dtype=float)

        stats[col]["missing"] += int(
            np.isnan(values).sum()
        )

        valid = values[~np.isnan(values)]

        if len(valid) > 0:
            stats[col]["min"] = min(
                stats[col]["min"], valid.min()
            )
            stats[col]["max"] = max(
                stats[col]["max"], valid.max()
            )
            stats[col]["sum"] += valid.sum()
            stats[col]["count"] += len(valid)

print("\nFEATURE AUDIT")
print("-" * 75)

for col, s in stats.items():
    mean = (
        s["sum"] / s["count"]
        if s["count"] else np.nan
    )

    print(f"\nFeature: {col}")
    print("Missing:", s["missing"])
    print("Minimum:", s["min"])
    print("Maximum:", s["max"])
    print("Mean:", mean)

print("\nFEATURE AUDIT COMPLETED")
