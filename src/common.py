from pathlib import Path

MODELS_PATH = Path("models")
DETECTOR_DATA_PATH = Path("data") / "detector"
CLASSIFIER_DATA_PATH = Path("data") / "classifier"
L2_REGULARIZATION = 1e-5
MOVING_AVERAGE_DECAY = 0.997

if not MODELS_PATH.exists():
    MODELS_PATH.mkdir(parents=True)


SCORE_MAP_THRESH = 0.5
NMS_TRESH = 0.1
BOX_THRESH = 0.05
