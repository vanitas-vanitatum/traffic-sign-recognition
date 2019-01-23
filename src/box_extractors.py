from typing import *

import numpy as np


class Extractor:
    def filter_out_empty_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        result = []
        for frame in frames:
            height, width = frame.shape[:-1]
            if height == 1 or width == 1 or height * width <= 1:
                continue
            result.append(frame)
        return result

    def extract(self, input_frame: np.ndarray, boxes: np.ndarray) -> List:
        raise NotImplementedError


class StandardExtractor(Extractor):
    def extract(self, input_frame: np.ndarray, boxes: np.ndarray) -> List:
        output_frames = []
        for x_min, y_min, x_max, y_max in boxes:
            frame = input_frame[y_min:y_max, x_min:x_max]
            output_frames.append(frame)
        return self.filter_out_empty_frames(output_frames)


EXTRACTORS = {
    "standard_extractor": StandardExtractor
}
