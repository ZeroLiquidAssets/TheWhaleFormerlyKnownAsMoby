from pathlib import Path



DATA_DIR = Path(__file__, "..", "data")

# Training 
TRAIN = DATA_DIR / "raw" / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "raw" / "train"

# Test
TEST_IMG_DIR = DATA_DIR / "raw" / "test"