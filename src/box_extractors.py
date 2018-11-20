import numpy as np
from typing import *


class Extractor:
    def extract(self, input_frame: np.ndarray, boxes: np.ndarray) -> List:
        raise NotImplementedError


class StandardExtractor(Extractor):
    def extract(self, input_frame: np.ndarray, boxes: np.ndarray) -> List:
        output_frames = []
        for x_min, y_min, x_max, y_max in boxes:
            frame = input_frame[y_min:y_max, x_min:x_max]
            output_frames.append(frame)
        return output_frames


EXTRACTORS = {
    "standard_extractor": StandardExtractor
}
