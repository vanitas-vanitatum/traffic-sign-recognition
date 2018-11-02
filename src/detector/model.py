import numpy as np
import tensorflow as tf

from common import L2_REGULARIZATION, MOVING_AVERAGE_DECAY
from detector.nets.pvanet import pvanet, unpool, relu_scale_convo


def predict_layer(inputs: tf.Tensor, features: int, kernel_size: int) -> tf.Tensor:
    x = tf.layers.conv2d(inputs, features, kernel_size, 1,
                         kernel_initializer=tf.keras.initializers.glorot_uniform(),
                         kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))
    x = tf.nn.sigmoid(x)
    return x


def normalize_images(images: tf.Tensor) -> tf.Tensor:
    return images / 255


def model(images: tf.Tensor, text_scale: int):
    images = normalize_images(images)

    end_points = pvanet(images)
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        f = [end_points['pool5'], end_points['pool4'],
             end_points['pool3'], end_points['pool2']]
        for i in range(4):
            tf.logging.info('Shape of f_{} {}'.format(i, f[i].shape))
        g = [None, None, None, None]
        h = [None, None, None, None]
        num_outputs = [None, 128, 64, 32]
        for i in range(4):
            if i == 0:
                h[i] = f[i]
            else:
                c1_1 = relu_scale_convo(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1, 1)
                h[i] = relu_scale_convo(c1_1, num_outputs[i], 3, 1)
            if i <= 2:
                g[i] = unpool(h[i])
            else:
                g[i] = relu_scale_convo(h[i], num_outputs[i], 3, 1)
            tf.logging.info('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

        f_score = predict_layer(g[3], 1, 1)
        geo_map = predict_layer(g[3], 4, 1) * text_scale
        angle_map = (predict_layer(g[3], 1, 1) - 0.5) * np.pi / 2
        f_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return f_score, f_geometry


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    """
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    """
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss


def model_fn(features, labels, mode, params):
    input_images = features["image"]
    input_score_maps = labels["score_maps"]
    input_geo_maps = labels["geo_maps"]
    input_training_masks = labels["training_masks"]
    text_scale = params["text_scale"]
    learning_rate = params["learning_rate"]

    f_score, f_geometry = model(input_images, text_scale)
    predictions = {
        "score": f_score,
        "geo_maps": f_geometry
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )
    else:
        model_loss = loss(input_score_maps, f_score,
                          input_geo_maps, f_geometry,
                          input_training_masks)
        total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000,
                                                   decay_rate=0.94, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate)

        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, tf.train.get_global_step())

        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = opt.minimize(total_loss, global_step=tf.train.get_global_step())
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op(name="train_op")

        tf.summary.image("input", input_images)
        tf.summary.image("score_map", input_score_maps)
        tf.summary.image("score_map_pred", f_score * 255)
        tf.summary.image("geo_map_0", input_geo_maps[:, :, :, 0:1])
        tf.summary.image("geo_map_0_pred", f_geometry[:, :, :, 0: 1])
        tf.summary.image("training_masks", input_training_masks)
        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("model_loss", model_loss)
        tf.summary.scalar("total_loss", total_loss)

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            predictions=predictions
        )
    return spec


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('text_scale', 512, '')
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()
