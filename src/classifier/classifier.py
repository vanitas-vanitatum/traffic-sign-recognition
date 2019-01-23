from typing import *

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util

from model import Model
from utils.math_utils import softmax


class Classifier(Model):
    def __init__(self, snapshot_path: str):
        super().__init__()
        self._graph = tf.Graph()
        self._snapshot_path = snapshot_path

        self._inputs: tf.Tensor = None
        self._output: tf.Tensor = None

        self.initiate()

    def initiate(self):
        with self._graph.as_default():
            op_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._snapshot_path, "rb") as fid:
                serialised_graph = fid.read()
                op_graph_def.ParseFromString(serialised_graph)
                tf.import_graph_def(op_graph_def, name="")
        self.sess = tf.Session(graph=self._graph)
        self._inputs = self._graph.get_tensor_by_name("input_images:0")
        self._output = self._graph.get_tensor_by_name("output:0")
        total_params = 0
        for node in self._graph.as_graph_def().node:
            if node.op == "Const":
                nd_array = tensor_util.MakeNdarray(node.attr['value'].tensor)
                total_params += np.prod(nd_array.shape, dtype=np.int32)

        print("Classifier params: {}".format(total_params))

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)
        outputs = self.sess.run(self._output, feed_dict={self._inputs: x})
        outputs_cls = np.argmax(outputs, axis=1)
        return outputs_cls, softmax(outputs)
