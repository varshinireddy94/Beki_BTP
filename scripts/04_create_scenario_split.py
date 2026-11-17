
import pandas as pd
from pathlib import Path

from config import DATA_PATH, OUTPUTS_DIR

# Load the available discharge scenarios
df = pd.read_parquet(DATA_PATH, columns=["Q"])
all_q = sorted(df["Q"].unique().tolist())

# Four completely unseen test discharges
test_q = [1200, 3600, 6800, 9200]

train_q = [q for q in all_q if q not in test_q]

assert len(all_q) == 48
assert len(train_q) == 44
assert len(test_q) == 4

# Save split information
split_df = pd.DataFrame({
    "Q": all_q,
    "split": [
        "test" if q in test_q else "train"
        for q in all_q
    ]
})

split_df.to_csv(
    OUTPUTS_DIR / "scenario_split.csv",
    index=False
)

print("Training scenarios:", len(train_q))
print("Test scenarios:", len(test_q))
print("Test Q values:", test_q)
print("Training Q range:", min(train_q), "-", max(train_q))
print("\nSplit saved successfully!")
