
from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import DATA_PATH, OUTPUTS_DIR

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration loaded successfully!")
print("Dataset:", DATA_PATH)
print("Dataset exists:", DATA_PATH.exists())

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")


# --------------------------------------------------
# 2. SCENARIO SPLIT
# --------------------------------------------------

VALIDATION_Q = [800, 3200, 6200, 8800]
TEST_Q = [1200, 3600, 6800, 9200]

CHUNK_SIZE = 200_000

TRAIN_PATH = OUTPUTS_DIR / "train_fhi.parquet"
VALIDATION_PATH = OUTPUTS_DIR / "validation_fhi.parquet"
TEST_PATH = OUTPUTS_DIR / "test_fhi.parquet"


# --------------------------------------------------
# 3. COLUMNS
# --------------------------------------------------

FEATURES = [
    "Q",
    "X",
    "Y",
    "bed_elevation",
    "NDWI_avg"
]

# Retain these for matching terrain slope later.
# They will NOT be used as model predictors.
IDENTIFIERS = ["row", "col"]

TARGET = "report_FHI"
FILTER_COLUMN = "report_eligible"

COLUMNS = (
    FEATURES
    + IDENTIFIERS
    + [TARGET, FILTER_COLUMN]
)


# --------------------------------------------------
# 4. READ AVAILABLE DISCHARGE SCENARIOS
# --------------------------------------------------

print("\nReading discharge scenarios...")

parquet_file = pq.ParquetFile(DATA_PATH)

all_q = set()

for batch in parquet_file.iter_batches(
    batch_size=CHUNK_SIZE,
    columns=["Q"]
):
    all_q.update(batch.column("Q").to_pylist())

all_q = sorted(all_q)

train_q = [
    q for q in all_q
    if q not in VALIDATION_Q and q not in TEST_Q
]

print("\nScenario split:")
print("Training scenarios:", len(train_q), train_q)
print("Validation scenarios:", len(VALIDATION_Q), VALIDATION_Q)
print("Test scenarios:", len(TEST_Q), TEST_Q)

assert len(train_q) == 40
assert len(VALIDATION_Q) == 4
assert len(TEST_Q) == 4

assert not set(train_q) & set(VALIDATION_Q)
assert not set(train_q) & set(TEST_Q)
assert not set(VALIDATION_Q) & set(TEST_Q)


# --------------------------------------------------
# 5. REMOVE OLD OUTPUT FILES
# --------------------------------------------------

for path in [TRAIN_PATH, VALIDATION_PATH, TEST_PATH]:
    if path.exists():
        path.unlink()


# --------------------------------------------------
# 6. CREATE PARQUET WRITERS
# --------------------------------------------------

writers = {
    "train": None,
    "validation": None,
    "test": None
}

output_paths = {
    "train": TRAIN_PATH,
    "validation": VALIDATION_PATH,
    "test": TEST_PATH
}

row_counts = {
    "train": 0,
    "validation": 0,
    "test": 0
}


# --------------------------------------------------
# 7. PROCESS DATA IN BATCHES
# --------------------------------------------------

print("\nPreparing FHI datasets...")

try:
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=CHUNK_SIZE,
            columns=COLUMNS
        ),
        start=1
    ):

        df = batch.to_pandas()

        # Only eligible land cells are used for FHI regression.
        df = df[
            df[FILTER_COLUMN].astype(bool)
        ]

        if df.empty:
            continue

        split_data = {
            "train": df[df["Q"].isin(train_q)],
            "validation": df[df["Q"].isin(VALIDATION_Q)],
            "test": df[df["Q"].isin(TEST_Q)]
        }

        for split_name, split_df in split_data.items():

            if split_df.empty:
                continue

            table = pa.Table.from_pandas(
                split_df,
                preserve_index=False
            )

            # Create writer for the first batch
            if writers[split_name] is None:
                writers[split_name] = pq.ParquetWriter(
                    output_paths[split_name],
                    table.schema,
                    compression="snappy"
                )

            writers[split_name].write_table(table)

            row_counts[split_name] += len(split_df)

        print(
            f"Processed batch {batch_number} | "
            f"Train: {row_counts['train']:,} | "
            f"Validation: {row_counts['validation']:,} | "
            f"Test: {row_counts['test']:,}"
        )

finally:
    # Always close writers, even if an error occurs
    for writer in writers.values():
        if writer is not None:
            writer.close()


# --------------------------------------------------
# 8. VERIFY OUTPUT DATASETS
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

for split_name, path in output_paths.items():

    if not path.exists():
        raise RuntimeError(
            f"{split_name} dataset was not created: {path}"
        )

    result = pd.read_parquet(path)

    print(f"\n{split_name.upper()}")
    print("File:", path)
    print("Shape:", result.shape)
    print("Discharges:", sorted(result["Q"].unique().tolist()))
    print("Columns:", result.columns.tolist())
    print("All rows eligible:", result[FILTER_COLUMN].all())

    assert "row" in result.columns
    assert "col" in result.columns
    assert TARGET in result.columns
    assert result[FILTER_COLUMN].all()

print("\nFinal row counts:")
print("Training:", f"{row_counts['train']:,}")
print("Validation:", f"{row_counts['validation']:,}")
print("Test:", f"{row_counts['test']:,}")

print("\nAll three datasets successfully saved and verified!")
