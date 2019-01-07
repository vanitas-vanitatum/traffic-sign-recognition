import numpy as np


def softmax(x, axis=1):
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
