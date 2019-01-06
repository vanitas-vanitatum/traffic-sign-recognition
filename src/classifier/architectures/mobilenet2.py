from typing import *

import tensorflow as tf
from tensorflow.keras.initializers import he_normal, glorot_uniform
from tensorflow.keras.regularizers import l2

from classifier.layers import spatial_dropout
from common import L2_REGULARIZATION, DROPOUT_VALUE


def convo_bn(inputs: tf.Tensor, features: int, kernel_size: int, is_training: bool, stride: int) -> tf.Tensor:
    x = tf.layers.conv2d(inputs, features, kernel_size, strides=(stride, stride), padding="same",
                         kernel_initializer=he_normal(), kernel_regularizer=l2(L2_REGULARIZATION))
    x = tf.layers.batch_normalization(x, training=is_training)
    return x


def depthwise_convo_bn(inputs: tf.Tensor, features: int, kernel_size: int, is_training: bool, stride: int) -> tf.Tensor:
    x = tf.layers.separable_conv2d(inputs, features, kernel_size, strides=(stride, stride), padding="same",
                                   depthwise_initializer=he_normal(), pointwise_initializer=he_normal(),
                                   depthwise_regularizer=l2(L2_REGULARIZATION),
                                   pointwise_regularizer=l2(L2_REGULARIZATION))
    x = tf.layers.batch_normalization(x, training=is_training)
    return x


def convo_bn_relu(inputs: tf.Tensor, features: int, kernel_size: int, is_training: bool, stride: int) -> tf.Tensor:
    x = convo_bn(inputs, features, kernel_size, is_training, stride)
    x = tf.nn.relu6(x)

    return x


def depthwise_bn_relu(inputs: tf.Tensor, features: int, kernel_size: int, is_training: bool, stride: int) -> tf.Tensor:
    x = depthwise_convo_bn(inputs, features, kernel_size, is_training, stride)
    x = spatial_dropout(x, drop_proba=DROPOUT_VALUE, is_training=is_training)
    x = tf.nn.relu6(x)
    return x


def bottleneck_residual_block(inputs: tf.Tensor, t_multiplier: int, output_channels: int, is_training: bool,
                              stride: int) -> tf.Tensor:
    input_channels = inputs.get_shape().as_list()[-1]
    x = convo_bn_relu(inputs, t_multiplier * input_channels, 1, is_training, 1)
    x = depthwise_bn_relu(x, t_multiplier * input_channels, 3, is_training, stride)
    x = tf.layers.conv2d(x, output_channels, 1, kernel_initializer=he_normal(),
                         kernel_regularizer=l2(L2_REGULARIZATION),
                         use_bias=True)
    if stride > 1:
        return x
    if input_channels != output_channels:
        inputs = tf.layers.conv2d(x, output_channels, 1, kernel_initializer=he_normal(),
                                  kernel_regularizer=l2(L2_REGULARIZATION))
        inputs = tf.layers.batch_normalization(inputs, training=is_training)
    return x + inputs


def construct_model(inputs: tf.Tensor, is_training: bool, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    strides = [1, 2, 2, 2, 1, 2, 1]
    channels = [16, 24, 32, 64, 96, 160, 320]
    repetitions = [1, 2, 3, 4, 3, 3, 1]
    multipliers = [1, 6, 6, 6, 6, 6, 6]

    inputs /= 255

    x = convo_bn_relu(inputs, 32, 3, is_training, 2)
    for i in range(len(channels)):
        for repetition_num in range(repetitions[i]):
            stride = min(int(repetition_num == 0) + 1, strides[i])
            x = bottleneck_residual_block(x, multipliers[i], output_channels=channels[i],
                                          is_training=is_training, stride=stride)

    x = tf.layers.conv2d(x, 1280, 1, kernel_initializer=he_normal(), kernel_regularizer=l2(L2_REGULARIZATION))
    x = tf.reduce_mean(x, axis=[1, 2])
    sign_class = tf.layers.dense(x, num_classes, kernel_initializer=glorot_uniform(),
                                 kernel_regularizer=l2(L2_REGULARIZATION), name="class")
    return sign_class
