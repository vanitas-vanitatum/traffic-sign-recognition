import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util

import common
from detector import lanms
from detector.icdar import restore_rectangle
from model import Model
from utils.graph_utils import load_graph


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
        self._graph: tf.Graph = load_graph(self._snapshot_path)
        self.sess = tf.Session(graph=self._graph)

        self._inputs = self._graph.get_tensor_by_name("import/input_images:0")
        self._f_score = self._graph.get_tensor_by_name("import/feature_fusion/score:0")
        self._f_geometry = self._graph.get_tensor_by_name("import/feature_fusion/geometry:0")

        total_params = 0
        for node in self._graph.as_graph_def().node:
            if node.op == "Const":
                nd_array = tensor_util.MakeNdarray(node.attr['value'].tensor)
                total_params += np.prod(nd_array.shape, dtype=np.int32)

        print("Detector params: {}".format(total_params))

    def resize_image(self, im, max_side_len=512):
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
        original_h, original_w, _ = x.shape
        img = x
        im_resized, (ratio_h, ratio_w) = self.resize_image(img, max_side_len=1280)
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
            final_boxes.append(box)
        final_boxes = np.asarray(final_boxes)
        x_mins = np.min(final_boxes[:, :, 0], axis=1)
        y_mins = np.min(final_boxes[:, :, 1], axis=1)
        x_maxes = np.max(final_boxes[:, :, 0], axis=1)
        y_maxes = np.max(final_boxes[:, :, 1], axis=1)

        final_final_boxes = np.stack((x_mins, y_mins, x_maxes, y_maxes), axis=-1).astype(np.int32)
        return final_final_boxes
