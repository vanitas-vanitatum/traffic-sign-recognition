import pickle as pkl
import time
from pathlib import Path
from typing import *

FRAME_HEIGHT = 768

MODELS_PATH = Path("../models")
SRC_PATH = Path(".")
FONT_PATH = Path('FiraMono-Medium.otf')
DETECTOR_DATA_PATH = Path("../data") / "detector"
CLASSIFIER_DATA_PATH = Path("../data") / "classifier"
CONFIG_PATH = Path("config.yml")
L2_REGULARIZATION = 1e-5
DROPOUT_VALUE = 0.3
MOVING_AVERAGE_DECAY = 0.997

if not MODELS_PATH.exists():
    MODELS_PATH.mkdir(parents=True)

CLASSIFIER_INPUT_WIDTH = 32
CLASSIFIER_INPUT_HEIGHT = 32
CLASSIFIER_THRESHOLD = 0.7
NO_SIGN_CLASS = "not_a_sign"

SCORE_MAP_THRESH = 0.5
NMS_TRESH = 0.2
BOX_THRESH = 0.1


def unpickle_data(path: str) -> Any:
    with open(path, 'rb') as f:
        return pkl.load(f)


class YoloConfig:
    input_shape_size = 416
    yolo_anchors_path = Path("detector") / "yolo" / "model" / "yolo_anchors.txt"


class TimerIt:
    def __init__(self):
        self._start = 0
        self._end = 0

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.time()
        print("Time spent: {}".format(self._end - self._start))
