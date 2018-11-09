import numpy as np


class Model:
    def __init__(self):
        self.sess = None

    def initiate(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
