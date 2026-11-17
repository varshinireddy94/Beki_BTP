
import pyarrow.parquet as pq
from config import DATA_PATH

print("\nChecking Beki ML dataset...\n")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

parquet_file = pq.ParquetFile(DATA_PATH)

print("Dataset loaded successfully!")
print("Total rows:", f"{parquet_file.metadata.num_rows:,}")
print("Total columns:", parquet_file.metadata.num_columns)

print("\nColumn names:")
print(parquet_file.schema.names)

print("\nDataset connection verified!")
