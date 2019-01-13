from typing import *

import tensorflow as tf
from tensorflow.keras.initializers import he_normal, glorot_uniform
from tensorflow.keras.regularizers import l2

from common import L2_REGULARIZATION

weight_regularizer = 0.0005


def convo_bn(inputs: tf.Tensor, features: int, kernel_size: int, is_training: bool, stride: int) -> tf.Tensor:
    x = tf.layers.conv2d(inputs, features, kernel_size, strides=(stride, stride), padding="same", use_bias=False,
                         kernel_initializer=he_normal(), kernel_regularizer=l2(weight_regularizer))
    x = tf.layers.batch_normalization(x, training=is_training)
    return x


def initial_convo(inputs: tf.Tensor, is_training: bool) -> tf.Tensor:
    x = convo_bn(inputs, 16, 3, is_training, 1)
    x = tf.nn.relu(x)
    return x


def expand_convo(inputs: tf.Tensor, filters: int, multiplier: int, is_training: bool, stride: int) -> tf.Tensor:
    x = convo_bn(inputs, filters * multiplier, 3, is_training, stride)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters * multiplier, 3, strides=(1, 1), padding="same", use_bias=False,
                         kernel_initializer=he_normal(), kernel_regularizer=l2(weight_regularizer))

    skip = tf.layers.conv2d(inputs, filters * multiplier, 1, strides=(stride, stride), padding="same", use_bias=False,
                            kernel_initializer=he_normal(), kernel_regularizer=l2(weight_regularizer))
    return x + skip


def rest_convo(inputs: tf.Tensor, filters: int, multiplier: int, is_training: bool) -> tf.Tensor:
    shape = tf.shape(inputs)
    x = convo_bn(inputs, filters * multiplier, 3, is_training, 1)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=0.1, noise_shape=[shape[0], 1, 1, shape[-1]])
    x = convo_bn(x, filters * multiplier, 3, is_training, 1)
    x = tf.nn.relu(x)
    return x + inputs


def construct_model(inputs: tf.Tensor, is_training: bool, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    inputs /= 255

    depth = 16
    n = (depth - 4) // 6
    multiplier = 4

    x = initial_convo(inputs, is_training)
    x = expand_convo(x, 16, multiplier, is_training, 1)
    for i in range(n - 1):
        x = rest_convo(x, 16, multiplier, is_training)

    x = expand_convo(x, 32, multiplier, is_training, 2)
    for i in range(n - 1):
        x = rest_convo(x, 32, multiplier, is_training)

    x = expand_convo(x, 64, multiplier, is_training, 2)
    for i in range(n - 1):
        x = rest_convo(x, 64, multiplier, is_training)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)

    x = tf.reduce_mean(x, axis=[1, 2])
    sign_class = tf.layers.dense(x, num_classes, kernel_initializer=glorot_uniform(),
                                 kernel_regularizer=l2(L2_REGULARIZATION), name="class")
    sign_class = tf.add(sign_class, 0, name="output")
    return sign_class
