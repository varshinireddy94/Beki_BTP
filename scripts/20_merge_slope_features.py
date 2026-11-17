
from pathlib import Path
import sys

import pandas as pd

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR

SLOPE_PATH = OUTPUTS_DIR / "terrain_features_q200.csv"

DATASETS = {
    "train": OUTPUTS_DIR / "train_fhi.parquet",
    "validation": OUTPUTS_DIR / "validation_fhi.parquet",
    "test": OUTPUTS_DIR / "test_fhi.parquet"
}

OUTPUTS = {
    "train": OUTPUTS_DIR / "train_fhi_with_slope.parquet",
    "validation": OUTPUTS_DIR / "validation_fhi_with_slope.parquet",
    "test": OUTPUTS_DIR / "test_fhi_with_slope.parquet"
}

print("=" * 60)
print("MERGING TERRAIN SLOPE FEATURES")
print("=" * 60)


# --------------------------------------------------
# 2. LOAD SLOPE DATA
# --------------------------------------------------

if not SLOPE_PATH.exists():
    raise FileNotFoundError(f"Slope file not found: {SLOPE_PATH}")

slope_df = pd.read_csv(SLOPE_PATH)

required_slope_columns = [
    "row",
    "col",
    "slope_degrees"
]

if not all(c in slope_df.columns for c in required_slope_columns):
    raise ValueError(
        f"Missing slope columns. Found: {slope_df.columns.tolist()}"
    )

slope_df = slope_df[required_slope_columns].copy()

print("\nSlope dataset:")
print("Shape:", slope_df.shape)
print("Columns:", slope_df.columns.tolist())

# Check missing values
if slope_df.isna().any().any():
    raise ValueError("Missing values found in slope dataset!")

# Check duplicate grid cells
if slope_df.duplicated(subset=["row", "col"]).any():
    raise ValueError("Duplicate row-col combinations in slope dataset!")

if len(slope_df) != 171600:
    raise ValueError(
        f"Expected 171600 grid cells, found {len(slope_df)}"
    )

print("Grid-cell uniqueness: PASSED")
print("Missing values: NONE")


# --------------------------------------------------
# 3. MERGE SLOPE INTO EACH DATASET
# --------------------------------------------------

for split_name, input_path in DATASETS.items():

    print("\n" + "-" * 50)
    print(f"Processing {split_name.upper()} dataset")

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_parquet(input_path)

    original_rows = len(df)

    print("Original shape:", df.shape)

    # Ensure identifiers are present
    if not {"row", "col"}.issubset(df.columns):
        raise ValueError(
            f"row/col columns missing in {split_name} dataset"
        )

    # Merge using exact grid identifiers
    merged_df = df.merge(
        slope_df,
        on=["row", "col"],
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------
    # 4. VALIDATION
    # --------------------------------------------------

    if len(merged_df) != original_rows:
        raise ValueError(
            f"Row count changed during {split_name} merge!"
        )

    missing_slope = merged_df["slope_degrees"].isna().sum()

    if missing_slope > 0:
        raise ValueError(
            f"{missing_slope} cells have missing slope values "
            f"in {split_name} dataset!"
        )

    print("Merged shape:", merged_df.shape)
    print("Missing slope values:", missing_slope)
    print("Row count preserved: YES")

    # --------------------------------------------------
    # 5. SAVE NEW DATASET
    # --------------------------------------------------

    output_path = OUTPUTS[split_name]

    merged_df.to_parquet(
        output_path,
        index=False,
        compression="snappy"
    )

    print("Saved:", output_path)

    # Release memory before next dataset
    del df, merged_df


# --------------------------------------------------
# 6. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("SLOPE MERGING COMPLETED SUCCESSFULLY!")
print("=" * 60)

for split_name, output_path in OUTPUTS.items():
    check_df = pd.read_parquet(output_path)

    print(f"\n{split_name.upper()}")
    print("Shape:", check_df.shape)
    print("Slope minimum:", check_df["slope_degrees"].min())
    print("Slope maximum:", check_df["slope_degrees"].max())
    print("Slope mean:", check_df["slope_degrees"].mean())

    assert "slope_degrees" in check_df.columns
    assert check_df["slope_degrees"].notna().all()

print("\nAll three datasets verified!")
