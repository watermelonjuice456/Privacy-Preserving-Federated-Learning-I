from pathlib import Path
APP_ROOT = Path(__file__).parents[1]

MODEL_SAVE_FILE = "demo_model.h5"
PARAMS_SAVE_FILE = "params.txt"
CIPHERTEXT_SAVE_FILE = "cipher.txt"
PUBLIC_KEY_SAVE_FILE = "public_key.pub"
TRAIN_LOCAL_MODEL = "/train"
AGGREGATE_VALUE = "/agg_val"

WORKER1_IP = "http://localhost:7101"
WORKER2_IP = "http://localhost:7102"
WORKER3_IP = "http://localhost:7103"
AGGREGATOR_IP = "http://localhost:7200"
