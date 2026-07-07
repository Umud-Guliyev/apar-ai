MODEL_NAME = "distilbert-base-multilingual-cased"

TRAIN_FILE = "data/train.csv"

TEXT_COLUMN = "feedback"
LABEL_COLUMN = "label"

MAX_LENGTH = 128

BATCH_SIZE = 16
EPOCHS = 5

LEARNING_RATE = 2e-5

NUM_LABELS = 3

SEED = 42

PATIENCE = 2

WARMUP_RATIO = 0.1