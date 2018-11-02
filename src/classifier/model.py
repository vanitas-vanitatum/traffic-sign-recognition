import numpy as np
import tensorflow as tf

import common


def convo_bn(inputs: tf.Tensor, features: int, kernel_size: int, is_training: bool, stride: int) -> tf.Tensor:
    x = tf.layers.conv2d(inputs, features, kernel_size, strides=(stride, stride), padding="same",
                         kernel_initializer=tf.keras.initializers.he_normal())
    x = tf.layers.batch_normalization(x, training=is_training)
    return x


def convo_bn_relu(inputs: tf.Tensor, features: int, kernel_size: int, is_training: bool, stride: int) -> tf.Tensor:
    x = convo_bn(inputs, features, kernel_size, is_training, stride)
    x = tf.nn.relu(x)

    return x


def initial_block(inputs: tf.Tensor, features: int, is_training: bool) -> tf.Tensor:
    x = tf.layers.conv2d(inputs, features, 3, strides=(1, 1), padding="same",
                         kernel_initializer=tf.keras.initializers.he_normal(),
                         kernel_regularizer=tf.keras.regularizers.l2(common.L2_REGULARIZATION))
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    return x


def projection_convo(inputs: tf.Tensor, features: int, stride: int):
    return tf.layers.conv2d(inputs, features, 1, strides=(stride, stride),
                            kernel_initializer=tf.keras.initializers.he_normal())


def depthwise_convo_bn(inputs: tf.Tensor, features: int, is_training: bool, stride: int) -> tf.Tensor:
    x = tf.layers.separable_conv2d(inputs, features, 3, strides=(stride, stride), padding="same",
                                   depthwise_initializer=tf.keras.initializers.he_normal(),
                                   pointwise_initializer=tf.keras.initializers.he_normal(),
                                   depthwise_regularizer=tf.keras.regularizers.l2(common.L2_REGULARIZATION),
                                   pointwise_regularizer=tf.keras.regularizers.l2(common.L2_REGULARIZATION))
    x = tf.layers.batch_normalization(x, training=is_training)

    return x


def se_module(inputs: tf.Tensor, inner_features: int) -> tf.Tensor:
    input_features = inputs.get_shape().as_list()[-1]

    x = tf.reduce_mean(inputs, axis=[1, 2])
    x = tf.layers.dense(x, inner_features, kernel_initializer=tf.keras.initializers.he_normal(),
                        kernel_regularizer=tf.keras.regularizers.l2(common.L2_REGULARIZATION))
    x = tf.nn.relu(x)
    x = tf.layers.dense(x, input_features, kernel_initializer=tf.keras.initializers.he_normal(),
                        kernel_regularizer=tf.keras.regularizers.l2(common.L2_REGULARIZATION))
    x = tf.nn.sigmoid(x)
    x = tf.reshape(x, (-1, 1, 1, input_features))
    return x * inputs


def shuffle_block(inputs: tf.Tensor, features: int, is_training: bool, stride: int, groups: int = 8) -> tf.Tensor:
    if stride > 1:
        first_branch, second_branch = inputs, inputs
    else:
        first_branch, second_branch = tf.split(inputs, num_or_size_splits=2, axis=-1)
    input_features = inputs.get_shape().as_list()[-1]

    if stride > 1:
        first_branch = depthwise_convo_bn(first_branch, features, is_training, stride)
        first_branch = convo_bn_relu(first_branch, features, 1, is_training, 1)

    second_branch = convo_bn_relu(second_branch, features, 1, is_training, 1)
    second_branch = depthwise_convo_bn(second_branch, features, is_training, stride)
    if stride == 1:
        second_branch = convo_bn(second_branch, features, 1, is_training, 1)
    else:
        second_branch = convo_bn_relu(second_branch, features, 1, is_training, 1)
    second_branch = se_module(second_branch, features // 2)

    if stride == 1:
        if input_features != features:
            inputs = projection_convo(inputs, features, 1)
        second_branch = inputs + second_branch
        second_branch = tf.nn.relu(second_branch)

    res = tf.concat([first_branch, second_branch], axis=-1)
    feats = res.get_shape().as_list()[-1]
    input_shape = tf.shape(res)

    res = tf.reshape(res,
                     (input_shape[0], input_shape[1], input_shape[2], groups, feats // groups))
    res = tf.transpose(res, perm=[0, 1, 2, 4, 3])
    res = tf.reshape(res, (input_shape[0], input_shape[1], input_shape[2], feats))
    return res


def model_fn(features, labels, mode, params):
    num_classes = params["num_classes"]
    learning_rate = params.get("learning_rate", 0.001)

    labels = labels["label"]
    net = features["image"]

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    net = initial_block(net, 24, is_training)

    repetitions = [4, 8, 4]
    features = [48, 96, 192]
    for rep, fs in zip(repetitions, features):
        for i in range(rep):
            net = shuffle_block(net, fs, is_training, int(i == 0) + 1)

    net = convo_bn_relu(net, 1024, 1, is_training, 1)

    net = tf.reduce_mean(net, axis=[1, 2])
    net = tf.layers.dense(net, num_classes,
                          kernel_initializer=tf.keras.initializers.he_normal(),
                          kernel_regularizer=tf.keras.regularizers.l2(common.L2_REGULARIZATION))

    logits = net
    y_pred = tf.nn.softmax(logits=logits)

    y_pred_cls = tf.argmax(y_pred, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels)
        loss = tf.reduce_mean(cross_entropy)

        total_loss = tf.add_n([loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        optimizer = tf.train.AdamOptimizer(learning_rate)

        variable_averages = tf.train.ExponentialMovingAverage(
            common.MOVING_AVERAGE_DECAY, tf.train.get_global_step())

        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(
                loss=total_loss, global_step=tf.train.get_global_step()
            )

        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op(name="train_op")

        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls),
            "precision": tf.metrics.precision(labels, y_pred_cls),
            "recall": tf.metrics.recall(labels, y_pred_cls),
            "false_positives": tf.metrics.false_positives(labels, y_pred_cls),
            "false_negatives": tf.metrics.false_positives(labels, y_pred_cls)
        }

        tf.summary.histogram("predictions", y_pred)
        tf.summary.scalar("unregularised_loss", loss)
        tf.summary.scalar("learning_rate", learning_rate)

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            predictions=y_pred_cls,
            eval_metric_ops=metrics
        )

    tf.logging.info("ShuffleNet params: {}".format(np.prod(sum([
        np.prod(var.get_shape().as_list()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    ]))))

    return spec
