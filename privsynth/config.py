import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_SYNTHETIC_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')

CTGAN_DEFAULT_CONFIG = {
    "epochs": 25,
    "batch_size": 500,
    "embedding_dim": 256,
    "seed": 42
}
