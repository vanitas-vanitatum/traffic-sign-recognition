import numpy as np
import tensorflow as tf

import common
from model import Model


class Detector(Model):
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

        ops = self._graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['detection_boxes', 'detection_scores']:
            tensor_name = key + ":0"
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self._graph.get_tensor_by_name(tensor_name)
        self._inputs = self._graph.get_tensor_by_name("image_tensor:0")
        self._output = tensor_dict

    def predict(self, x: np.ndarray) -> np.ndarray:
        results = []
        for sample in x:
            results.append(self.predict_single_frame(sample))
        return np.asarray(results)

    def predict_single_frame(self, x: np.ndarray) -> np.ndarray:
        height, width = x.shape[:2]
        output_dict = self.sess.run(self._output, feed_dict={self._inputs: np.expand_dims(x, axis=0)})
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        boxes = [[box[1] * width, box[0] * height, box[3] * width, box[2] * height]
                 for box, score in zip(output_dict['detection_boxes'], output_dict['detection_scores'])
                 if score > common.SCORE_MAP_THRESH]
        return np.asarray(boxes, dtype=np.int32)
