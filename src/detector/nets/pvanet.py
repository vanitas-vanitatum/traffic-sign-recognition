from typing import *

import tensorflow as tf
from tensorflow.keras import initializers

from common import L2_REGULARIZATION

scaling_layer_counter = 1


def unpool(inputs: tf.Tensor) -> tf.Tensor:
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def scaling_layer(inputs: tf.Tensor) -> tf.Tensor:
    global scaling_layer_counter
    input_channels = inputs.get_shape().as_list()[-1]
    with tf.name_scope("scale_shift_{}".format(scaling_layer_counter)) as name_scope:
        scales = tf.get_variable(name_scope + "/scales", (input_channels,), tf.float32, initializer=initializers.Ones(),
                                 regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))
        shifts = tf.get_variable(name_scope + "/shifts", (input_channels,), tf.float32,
                                 initializer=initializers.Zeros(),
                                 regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))
        scaling_layer_counter += 1
        return inputs * scales + shifts


def concatenated_convo(inputs: tf.Tensor, features: int, kernel_size: int, stride: int) -> tf.Tensor:
    x = tf.layers.conv2d(inputs, features, kernel_size, stride, padding="same",
                         kernel_initializer=initializers.he_normal(),
                         kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))
    return tf.concat([x, -x], axis=-1)


def relu_scale_concatenated_convo(inputs: tf.Tensor, features: int, kernel_size: int, stride: int):
    x = scaling_layer(inputs)
    x = tf.nn.relu(x)
    x = concatenated_convo(x, features, kernel_size, stride)
    return x


def relu_scale_convo(inputs: tf.Tensor, features: int, kernel_size: int, stride: int):
    x = scaling_layer(inputs)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, features, kernel_size, stride, padding="same",
                         kernel_initializer=initializers.he_normal(),
                         kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))
    return x


def initial_block(inputs: tf.Tensor) -> tf.Tensor:
    return concatenated_convo(inputs, 16, 7, 2)


def crelu_block(inputs: tf.Tensor, inner_features: int, outer_features: int, kernel_size: int,
                stride: int) -> tf.Tensor:
    input_features = inputs.get_shape().as_list()[-1]
    x = relu_scale_convo(inputs, inner_features, 1, 1)
    x = relu_scale_concatenated_convo(x, inner_features, kernel_size, stride)
    x = relu_scale_convo(x, outer_features, 1, 1)

    if input_features != outer_features or stride > 1:
        inputs = relu_scale_convo(inputs, outer_features, 1, stride)
    return x + inputs


def inception_block(inputs: tf.Tensor, features_1x1: int, features_3x3: Sequence[int],
                    features_5x5: Sequence[int], pool: Optional[int], out: int, stride: int):
    b1 = relu_scale_convo(inputs, features_1x1, 1, stride)

    b2 = relu_scale_convo(inputs, features_3x3[0], 1, stride)
    b2 = relu_scale_convo(b2, features_3x3[1], 3, 1)

    b3 = relu_scale_convo(inputs, features_5x5[0], 1, stride)
    b3 = relu_scale_convo(b3, features_5x5[1], 3, 1)
    b3 = relu_scale_convo(b3, features_5x5[2], 3, 1)

    to_concat = [b1, b2, b3]

    if stride > 1:
        assert pool is not None
        b4 = tf.layers.max_pooling2d(inputs, (3, 3), (2, 2), padding="same")
        b4 = relu_scale_convo(b4, pool, 1, 1)
        to_concat.append(b4)

    result = tf.concat(to_concat, axis=-1)
    result = relu_scale_convo(result, out, 1, 1)
    return result


def pvanet(inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
    conv1_1 = initial_block(inputs)
    pool1_1 = tf.layers.max_pooling2d(conv1_1, (3, 3), (2, 2), padding="same")

    conv2_1 = crelu_block(pool1_1, 24, 64, 3, 1)
    conv2_2 = crelu_block(conv2_1, 24, 64, 3, 1)
    conv2_3 = crelu_block(conv2_2, 24, 64, 3, 1)

    conv3_1 = crelu_block(conv2_3, 48, 128, 3, 2)
    conv3_2 = crelu_block(conv3_1, 48, 128, 3, 1)
    conv3_3 = crelu_block(conv3_2, 48, 128, 3, 1)
    conv3_4 = crelu_block(conv3_3, 48, 128, 3, 1)

    conv4_1 = inception_block(conv3_4, 64, [48, 128], [24, 48, 48], 128, 256, 2)
    conv4_2 = inception_block(conv4_1, 64, [64, 128], [24, 48, 48], None, 256, 1)
    conv4_3 = inception_block(conv4_2, 64, [64, 128], [24, 48, 48], None, 256, 1)
    conv4_4 = inception_block(conv4_3, 64, [64, 128], [24, 48, 48], None, 256, 1)

    conv5_1 = inception_block(conv4_4, 64, [96, 192], [32, 64, 64], 128, 384, 2)
    conv5_2 = inception_block(conv5_1, 64, [96, 192], [32, 64, 64], None, 384, 1)
    conv5_3 = inception_block(conv5_2, 64, [96, 192], [32, 64, 64], None, 384, 1)
    conv5_4 = inception_block(conv5_3, 64, [96, 192], [32, 64, 64], None, 384, 1)

    return {
        'pool2': conv2_3,
        'pool3': conv3_4,
        'pool4': conv4_4,
        'pool5': conv5_4
    }
