from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = ROOT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Random seed
SEED = 42

# Validation
TEST_SIZE = 0.2

# TF-IDF
NGRAM_RANGE = (1, 1)
MIN_DF = 1
MAX_DF = 1.0

# Model
MODEL_NAME = "LinearSVC"