
import pandas as pd
import pyarrow.parquet as pq

from config import DATA_PATH

print("Reading dataset in batches...")

pf = pq.ParquetFile(DATA_PATH)

total_rows = 0
scenario_counts = {}
class_counts = {}
eligible_count = 0

for batch in pf.iter_batches(
    batch_size=200_000,
    columns=[
        "Q",
        "report_eligible",
        "report_hazard_class"
    ]
):
    df = batch.to_pandas()
    total_rows += len(df)

    # Count records per discharge
    q_counts = df["Q"].value_counts()
    for q, count in q_counts.items():
        scenario_counts[q] = (
            scenario_counts.get(q, 0) + count
        )

    # Count eligible cells
    eligible_count += int(
        df["report_eligible"].sum()
    )

    # Count hazard classes only for eligible cells
    eligible = df[df["report_eligible"]]

    counts = eligible["report_hazard_class"].value_counts()
    for cls, count in counts.items():
        class_counts[int(cls)] = (
            class_counts.get(int(cls), 0) + count
        )

print("\nDATASET SUMMARY")
print("Total records:", f"{total_rows:,}")
print("Number of discharges:", len(scenario_counts))
print("Eligible records:", f"{eligible_count:,}")

print("\nRECORDS PER DISCHARGE")
for q in sorted(scenario_counts):
    print(q, ":", f"{scenario_counts[q]:,}")

print("\nHAZARD CLASS DISTRIBUTION")
for cls in range(1, 6):
    count = class_counts.get(cls, 0)
    print(
        f"Class {cls}: {count:,} "
        f"({count / eligible_count * 100:.2f}%)"
    )

print("\nANALYSIS COMPLETED")
