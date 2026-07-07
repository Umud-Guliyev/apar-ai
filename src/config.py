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
NGRAM_RANGE = (3,5)
MIN_DF = 1
MAX_DF = 1.0
ANALYZER = "char_wb"


# Model
MODEL_NAME = "LinearSVC"

FEATURE_SET = "char"

WORD_NGRAM = (1,2)

CHAR_NGRAM = (3,5)

CLASS_WEIGHT = "balanced"