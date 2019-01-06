import io
import os
import pickle
import random
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import *

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder

random.seed(0)
np.random.seed(0)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def extract_class_from_img_path(path: Path) -> str:
    return path.parent.name


def rebalance_by_path(paths: List[Path]) -> List[Path]:
    classes = [extract_class_from_img_path(path) for path in paths]
    counts = Counter(classes)
    maximal_count = int(max([val for val in counts.values()]) * FLAGS.fraction)
    groups = groupby(paths, key=lambda x: extract_class_from_img_path(x))
    new_paths = []
    tf.logging.info("New count {} for {} classes".format(maximal_count, len(classes)))
    for key, group in groups:
        group = list(group)
        cls_count = counts[key]
        remained_paths_to_fill = maximal_count - cls_count
        tf.logging.info(f"Class: {key}, instances: {cls_count}, to fill: {remained_paths_to_fill}")
        if remained_paths_to_fill < 0:
            new_paths += np.random.choice(group, size=maximal_count, replace=False).tolist()
        else:
            new_paths += random.choices(group, k=remained_paths_to_fill) + group
        tf.logging.info(f"Current added instances after rebalancing class {key}: {len(new_paths)}")
    return new_paths


def create_tfrecord(image_path: Path, encoder: LabelEncoder):
    with tf.gfile.GFile(image_path.as_posix(), "rb") as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_io)
    width, height = image.size

    filename = image_path.as_posix().encode("utf-8")
    image_format = b"png"
    class_num = encoder.transform([image_path.parent.name])
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": int64_feature(height),
        "image/width": int64_feature(width),
        "image/filename": bytes_feature(filename),
        "image/encoded": bytes_feature(encoded_img),
        "image/format": bytes_feature(image_format),
        "image/class": int64_feature(class_num)
    }))
    return tf_example


def main(_):
    output_path = Path(FLAGS.output_path)
    input_path = Path(FLAGS.input_path)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    classes = os.listdir(FLAGS.input_path)
    if FLAGS.save_classes:
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        with open(output_path.parent / "label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(FLAGS.encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
    images = list(input_path.rglob("*.*"))
    if FLAGS.rebalance_classes:
        tf.logging.info("Rebalancing classes...")
        images = rebalance_by_path(images)
    random.shuffle(images)
    for img_file in tqdm.tqdm(images):
        tf_example = create_tfrecord(img_file, label_encoder)
        writer.write(tf_example.SerializeToString())
    writer.close()
    tf.logging.info("Successfully saved file: %s" % output_path.as_posix())


if __name__ == '__main__':
    tf.flags.DEFINE_string("input_path", "", "Input path for data")
    tf.flags.DEFINE_string("output_path", "", "Output path for tfrecord")
    tf.flags.DEFINE_string("encoder_path", "", "Path to object encoding classes, ie. LabelEncoder")
    tf.flags.DEFINE_bool("save_classes", False, "Whether save LabelEncoder object with class encoding")
    tf.flags.DEFINE_bool("rebalance_classes", False,
                         "Whether rebalance classes to maximal value existing in the dataset")
    tf.flags.DEFINE_float("fraction", 0.7, "Fraction of maximum samples per class times a count of the most numerous"
                                           "class")
    FLAGS = tf.flags.FLAGS
    tf.app.run()
