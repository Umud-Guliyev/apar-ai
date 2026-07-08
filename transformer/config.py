import os

MODEL_NAME = "xlm-roberta-base"

TRAIN_FILE = "data/train.csv"

TEXT_COLUMN = "feedback"
LABEL_COLUMN = "label"

MAX_LENGTH = 256

BATCH_SIZE = 16

EPOCHS = 8

LEARNING_RATE = 1e-5

NUM_LABELS = 3

SEED = 42

PATIENCE = 2

WARMUP_RATIO = 0.1


# ==========================
# Storage
# ==========================

BASE_DIR = "/content/drive/MyDrive/apar-ai"

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")