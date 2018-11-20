import os

import numpy as np
import tensorflow as tf
from PIL import Image

import common
from detector.yolo.detect_function import predict
from detector.yolo.network_function import construct_model
from model import Model


class Detector(Model):

    def __init__(self, snapshot_path: str):
        super().__init__()
        self._graph = tf.Graph()
        self._snapshot_path = snapshot_path
        self._anchors = None

        self._inputs: tf.Tensor = None
        self._image_shape: tf.Tensor = None
        self._output: tf.Tensor = None

        self.initiate()

    def initiate(self):
        with open(common.YoloConfig.yolo_anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)

        with self._graph.as_default():
            self._inputs = tf.placeholder(tf.float32, shape=[None, common.YoloConfig.input_shape_size,
                                                             common.YoloConfig.input_shape_size, 3])
            self._image_shape = tf.placeholder(tf.float32, shape=[2])
            scale1, scale2, scale3 = construct_model(self._inputs, 2)
            scale_total = [scale1, scale2, scale3]

            boxes, scores, classes = predict(scale_total, anchors, 2, self._image_shape,
                                             score_threshold=common.SCORE_MAP_THRESH,
                                             iou_threshold=common.SCORE_MAP_THRESH)
            saver = tf.train.Saver(tf.trainable_variables())

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            ckpt_state = tf.train.get_checkpoint_state(self._snapshot_path)
            model_path = os.path.join(self._snapshot_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(self.sess, model_path)

            self._output = boxes

    def predict(self, x: np.ndarray) -> np.ndarray:
        results = []
        for sample in x:
            results.append(self.predict_single_frame(sample))
        return np.asarray(results)

    def predict_single_frame(self, x: np.ndarray) -> np.ndarray:
        image = Image.fromarray(x)
        image_w, image_h = image.size
        image_shape = np.array([image_h, image_w])
        w, h = common.YoloConfig.input_shape_size, common.YoloConfig.input_shape_size
        new_w = int(image_w * min(w / image_w, h / image_h))
        new_h = int(image_h * min(w / image_w, h / image_h))
        resized_image = image.resize((new_w, new_h), Image.BICUBIC)

        boxed_image = Image.new('RGB', (w, h), (128, 128, 128))
        boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
        image_data = np.asarray(boxed_image, dtype="float32")

        image_data /= 255.
        inputs = np.expand_dims(image_data, 0)

        boxes = self.sess.run(self._output, feed_dict={
            self._inputs: inputs,
            self._image_shape: image_shape
        })

        out_boxes = []
        for y_min, x_min, y_max, x_max in boxes:
            out_box = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]
            out_boxes.append(out_box)
        if len(out_boxes) == 0:
            return np.asarray(out_boxes, dtype=np.int32)
        out_boxes = np.asarray(out_boxes, dtype=np.float32)
        # out_boxes[..., 0] = (out_boxes[..., 0] - offset_left) / ratio_w
        # out_boxes[..., 1] = (out_boxes[..., 1] - offset_top) / ratio_h
        out_boxes = out_boxes.astype(np.int32)

        return out_boxes
