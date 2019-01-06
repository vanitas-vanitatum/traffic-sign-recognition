import os
import pickle as pkl
from itertools import chain
from pathlib import Path
from typing import *

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import imutils
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from classifier.data_utils import rebalance_by_path
from common import CLASSIFIER_INPUT_HEIGHT, CLASSIFIER_INPUT_WIDTH

np.random.seed(0)


def get_images(data_path: Path) -> Iterable[Path]:
    return chain(
        data_path.rglob("*.png"),
        data_path.rglob("*.jpg"),
        data_path.rglob("*.tiff"),
        data_path.rglob("*.jpeg"),
    )


def get_augmenters() -> iaa.Sequential:
    return iaa.Sequential(children=[
        # iaa.JpegCompression((20, 90)),
        iaa.Add((-20, 20)),
        iaa.Multiply((0.8, 1.2)),
        iaa.AdditiveGaussianNoise(0, 0.1),
        iaa.Affine(
            # scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            # rotate=(-15, 15),
            mode=ia.ALL
        )], random_order=True
    )


class DataLoader:
    def __init__(self, data_path: str, is_training: bool):
        self._data_path = Path(data_path)
        self._is_training = is_training

        self.all_classes = self.get_classes_from_data_path()
        tf.logging.info("Loaded {} classes".format(self.all_classes))

        self.num_classes = len(self.all_classes)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.all_classes)
        self.indexed_classes = self.label_encoder.transform(self.all_classes)

        self._augmenters = get_augmenters() if is_training else None
        self._one_hot_class_helper = np.eye(self.num_classes)

        self._x_paths = list(get_images(self._data_path))
        if self._is_training:
            self._x_paths = rebalance_by_path(self._x_paths, fraction_of_most_counted_class=0.6)

        self._x_paths = np.asarray(self._x_paths)
        np.random.shuffle(self._x_paths)
        self._x_paths = np.asarray([str(x.as_posix()).encode('utf-8') for x in self._x_paths])

    def get_classes_from_data_path(self) -> List[str]:
        paths = self._data_path.iterdir()
        return [p.name for p in paths]

    def save_label_encoder(self, path: str):
        with open(path, 'wb') as f:
            pkl.dump(self.label_encoder, f, protocol=pkl.HIGHEST_PROTOCOL)

    def input_fn(self, batch_size: int = 32, buffer_size: int = 2048):
        def _f():
            return self._input_fn(batch_size, buffer_size)

        return _f

    def _prepare_single_record(self, path: bytes):
        img_path = path.decode('utf-8')
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        if height > width:
            img = imutils.resize(img, height=CLASSIFIER_INPUT_HEIGHT)
            to_pad_from_left = (32 - img.shape[1]) // 2
            to_pad_from_right = (32 - img.shape[1] - to_pad_from_left)
            padding = [[0, 0], [to_pad_from_left, to_pad_from_right], [0, 0]]
        else:
            img = imutils.resize(img, width=CLASSIFIER_INPUT_WIDTH)
            to_pad_from_top = (32 - img.shape[0]) // 2
            to_pad_from_bottom = (32 - img.shape[0] - to_pad_from_top)
            padding = [[to_pad_from_top, to_pad_from_bottom], [0, 0], [0, 0]]
        img = np.pad(img, padding, mode="constant", constant_values=0)
        if self._is_training:
            img = self._augmenters.augment_image(img)

        class_name = os.path.basename(os.path.dirname(img_path))
        a_class = self.label_encoder.transform([class_name])[0]
        img = img.astype(np.float32)
        a_class = a_class.astype(np.int32)
        return img, a_class

    def _process_input_tensor(self, input_tensor: tf.Tensor):
        img, a_class_one_hot = tf.py_func(self._prepare_single_record, inp=[input_tensor],
                                          Tout=[tf.float32, tf.int32])
        img.set_shape([None, None, 3])
        a_class_one_hot.set_shape([])
        return img, a_class_one_hot

    def _input_fn(self, batch_size: int, buffer_size: int = 2048) -> Iterable[Dict[str, tf.Tensor]]:
        dataset = tf.data.Dataset.from_tensor_slices(self._x_paths)
        if self._is_training:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.map(self._process_input_tensor, num_parallel_calls=8)
        dataset.repeat(1)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size)

        iterator = dataset.make_one_shot_iterator()
        images_batch, labels_batch = iterator.get_next()

        x = {
            "image": images_batch
        }
        y = {
            "label": labels_batch,
        }
        return x, y
