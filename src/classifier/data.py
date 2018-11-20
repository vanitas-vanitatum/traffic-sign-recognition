from functools import partial
from typing import *

import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf

from common import CLASSIFIER_INPUT_HEIGHT, CLASSIFIER_INPUT_WIDTH


def get_augmenters() -> iaa.Sequential:
    return iaa.Sequential(children=[
        iaa.JpegCompression((10, 90)),
        iaa.AdditiveGaussianNoise(0, 0.1),
        iaa.Affine(
            scale=(0.5, 1.5),
            translate_percent=(0.8, 1.2),
            rotate=(-15, 15),
            mode=ia.ALL
        )], random_order=True
    )


def parser(record, train, augmenters: Optional[iaa.Sequential] = None):
    keys_to_features = {
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/format": tf.FixedLenFeature([], tf.string),
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/class": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    shape = tf.stack([parsed["image/height"], parsed["image/width"], 3])
    image = tf.image.decode_image(parsed["image/encoded"], channels=3, dtype=tf.uint8)
    image = tf.reshape(image, shape)
    image = tf.image.resize_image_with_pad(image, CLASSIFIER_INPUT_HEIGHT, CLASSIFIER_INPUT_WIDTH)
    image = tf.image.rgb_to_grayscale(image)
    if train:
        image = tf.image.random_brightness(image, 0.4)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        # current_shape = image.shape
        # image = tf.py_func(augmenters.augment_image, [image], tf.uint8, stateful=False)
        # image.set_shape(current_shape)
    image = tf.cast(image, tf.float32)
    image = image / 255
    label = tf.cast(parsed["image/class"], tf.int32)
    return image, label


def _input_fn(file_path: str, batch_size: int, train: bool, buffer_size: int = 2048) -> Iterable[Dict[str, tf.Tensor]]:
    dataset = tf.data.TFRecordDataset(filenames=[file_path], buffer_size=buffer_size * 10, num_parallel_reads=2)
    parse = partial(parser, train=train, augmenters=get_augmenters())
    dataset = dataset.map(parse, 8)
    if train:
        dataset = dataset.shuffle(buffer_size)
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


def train_input_fn(file_path: str, batch_size: int = 32):
    def _f():
        return _input_fn(file_path, batch_size, True)

    return _f


def test_input_fn(file_path: str, batch_size: int = 32):
    def _f():
        return _input_fn(file_path, batch_size, False)

    return _f
