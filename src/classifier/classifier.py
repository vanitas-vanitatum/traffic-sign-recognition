import numpy as np
import tensorflow as tf

from model import Model


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

    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.asarray([], dtype=np.int32)
        outputs = self.sess.run(self._output, feed_dict={self._inputs: x})
        outputs = np.argmax(outputs, axis=1)
        return outputs
