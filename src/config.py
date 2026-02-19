from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ZIP_PATH = ROOT/"data"/"raw"/"CMAPSSdata.zip"
DATA_RAW = ROOT/"data"/"raw"
DATA_PROCESSED = ROOT /"data"/"processed"

OUTPUTS_PLOTS = ROOT/"outputs"/"plots"
OUTPUTS_MODELS = ROOT/"outputs"/"models"

DATASET = "FD001"

CLIP_RUL = 120
VAR_THRESHOLD= 1e-5
WINDOW = 5
CORR_THRESHOLD = 0.1

TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_FEATURES = 15
N_SPLITS = 3

THRESHOLD = 13