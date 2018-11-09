import os
import time

import cv2
import numpy as np
import tensorflow as tf

import common
from detector import lanms
from detector.icdar import restore_rectangle
from detector.model import model
from model import Model


class Detector(Model):
    def __init__(self, snapshot_path: str):
        super().__init__()
        self._graph = tf.Graph()
        self._snapshot_path = snapshot_path

        self._inputs: tf.Tensor = None

        self._f_score: tf.Tensor = None
        self._f_geometry: tf.Tensor = None

        self.initiate()

    def initiate(self):
        with self._graph.as_default():
            self._inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
            global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0),
                                          trainable=False)

            self._f_score, self._f_geometry = model(self._inputs, 512, is_training=True)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())
            self.sess = tf.Session(graph=self._graph)
            ckpt_state = tf.train.get_checkpoint_state(self._snapshot_path)
            model_path = os.path.join(self._snapshot_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(self.sess, model_path)

    def resize_image(self, im, max_side_len=2400):
        """
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        im = cv2.resize(im, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)

    def detect(self, score_map, geo_map, score_map_thresh=common.SCORE_MAP_THRESH,
               box_thresh=common.BOX_THRESH, nms_thres=common.NMS_TRESH):
        """
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        """
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        # nms part
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

        if boxes.shape[0] == 0:
            return boxes

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes

    def sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def predict(self, x: np.ndarray) -> np.ndarray:
        results = []
        for sample in x:
            results.append(self.predict_single_frame(sample))
        return np.asarray(results)

    def predict_single_frame(self, x: np.ndarray) -> np.ndarray:
        img = x
        im_resized, (ratio_h, ratio_w) = self.resize_image(img)
        score, geometry = self.sess.run(
            [self._f_score, self._f_geometry],
            feed_dict={self._inputs: [im_resized]})

        boxes = self.detect(score_map=score, geo_map=geometry)
        if boxes.shape[0] == 0:
            return boxes
        scores = boxes[:, 8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

        final_boxes = []
        for box, score in zip(boxes, scores):
            box = self.sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            final_boxes.append(box)
        return np.asarray(final_boxes)
