import os

basedir = os.path.abspath(os.path.dirname(__file__))
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKER_IP = "0.0.0.0"

PARAMS_FILE_ENDPOINT = "/get_saved_params"
PARAMS_JSON_ENDPOINT = "/get_params"
SERVER_MODEL_ENDPOINT = "/get_model"
SERVER_WEIGHT_ENDPOINT = "/get_model_weights"
SERVER_GET_PUBLIC_KEY_ENDPOINT = "/get_public_key"
SAVE_WEIGHT_MATRIX_ENDPOINT = "/save_weights"
PARAMS_SAVE_FILE = "params.txt"
MODEL_SAVE_FILE = "model.h5"
CIPHERTEXT_SAVE_FILE = "cipher.txt"
PUBLIC_KEY_SAVE_FILE = "public_key.pub"


class DefaultConfig:
    """Base config."""
    SERVER_IP = "http://localhost:7000"
    AGGREGATOR_IP = "http://localhost:7200"


class DockerConfig(DefaultConfig):
    SERVER_IP = "http://pyseal-demo-server:7000"
    AGGREGATOR_IP = "http://pyseal-demo-aggregator-service:7200"
