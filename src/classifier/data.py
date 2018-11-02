import pickle as pkl
from pathlib import Path
from typing import *

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class DataGenerator:
    def __init__(self, path: Path, seed: Optional[int] = None):
        self._path = path
        self._le = LabelEncoder()
        self._rng = np.random.RandomState(seed)

        self._augs = self._augmenters()
        self._imgs_paths, self._classes, self._countries = self._load_data()

    def _preprocess_img(self, img: np.ndarray) -> np.ndarray:
        img = self._augs.augment_image(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32)
        img /= 255
        img = img[..., np.newaxis]
        return img

    def _augmenters(self) -> iaa.Sequential:
        return iaa.Sequential([
            iaa.JpegCompression((20, 80))
        ])

    def _get_generator(self) -> Tuple[np.ndarray, np.ndarray]:
        while True:
            for img_path, cls in zip(self._imgs_paths, self._classes):
                img = cv2.imread(img_path)[..., ::-1]
                img = cv2.resize(img, (32, 32))
                img = self._preprocess_img(img)
                cls = cls.astype(np.int32)
                yield img, cls

    def _input_fn(self, batch_size: int, train: bool, buffer_size: int = 2048) -> Iterable[Dict[str, tf.Tensor]]:
        dataset = tf.data.Dataset.from_generator(generator=self._get_generator,
                                                 output_shapes=(
                                                     tf.TensorShape([32, 32, 1]),
                                                     tf.TensorShape([]),
                                                 ),
                                                 output_types=(tf.float32, tf.int32,))
        if train:
            dataset = dataset.shuffle(buffer_size)
        dataset.repeat(1)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        images_batch, labels_batch = iterator.get_next()

        x = {
            "image": images_batch
        }
        y = {
            "label": labels_batch,
        }
        return x, y

    def train_input_fn(self, batch_size: int = 32):
        def _f():
            return self._input_fn(batch_size, True)

        return _f

    def test_input_fn(self, batch_size: int = 32):
        def _f():
            return self._input_fn(batch_size, False)

        return _f

    def dump_label_encoder(self, path: Path):
        with open(path.as_posix(), "wb") as f:
            pkl.dump(self._le, f, protocol=pkl.HIGHEST_PROTOCOL)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_info = pd.read_csv(self._path.as_posix())
        imgs_paths = data_info["filename"].values
        imgs_paths = np.asarray([
            str((Path(self._path.parent) / path).as_posix()) for path in imgs_paths
        ])

        classes = data_info["class"].values
        country = data_info["country"].values

        self._le.fit(classes)
        classes = self._le.transform(classes)
        return imgs_paths, classes, country
