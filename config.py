
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Existing dataset location
DATA_PATH = Path(
    r"E:\btp files\Beki_ML_Dataset_48Q.parquet"
)

# Project output directories
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
APP_DIR = PROJECT_DIR / "app"

# Create directories if they don't exist
for folder in [MODELS_DIR, OUTPUTS_DIR, APP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("Configuration loaded successfully!")
print("Dataset:", DATA_PATH)
print("Dataset exists:", DATA_PATH.exists())
