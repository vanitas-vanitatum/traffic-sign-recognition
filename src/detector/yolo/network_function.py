# MODEL_NETWORK
from typing import *

import numpy as np
import tensorflow as tf

from detector.yolo.pretrained_model_loader import W, B


def construct_model(inputs: tf.Tensor, num_classes: int) -> Tuple[tf.Tensor, ...]:
    with tf.name_scope("Features"):
        conv_1 = conv2d(inputs, 1, num_classes)
        conv_2 = conv2d(conv_1, 2, num_classes, stride=2)

        conv_3 = conv2d(conv_2, 3, num_classes)
        conv_4 = conv2d(conv_3, 4, num_classes)
        resn_1 = resnet(conv_2, conv_4, 1)
        conv_5 = conv2d(resn_1, 5, num_classes, stride=2)

        conv_6 = conv2d(conv_5, 6, num_classes)
        conv_7 = conv2d(conv_6, 7, num_classes)
        resn_2 = resnet(conv_5, conv_7, 2)

        conv_8 = conv2d(resn_2, 8, num_classes)
        conv_9 = conv2d(conv_8, 9, num_classes)
        resn_3 = resnet(resn_2, conv_9, 3)
        conv_10 = conv2d(resn_3, 10, num_classes, stride=2)

        conv_11 = conv2d(conv_10, 11, num_classes)
        conv_12 = conv2d(conv_11, 12, num_classes)
        resn_4 = resnet(conv_10, conv_12, 4)

        conv_13 = conv2d(resn_4, 13, num_classes)
        conv_14 = conv2d(conv_13, 14, num_classes)
        resn_5 = resnet(resn_4, conv_14, 5)

        conv_15 = conv2d(resn_5, 15, num_classes)
        conv_16 = conv2d(conv_15, 16, num_classes)
        resn_6 = resnet(resn_5, conv_16, 6)

        conv_17 = conv2d(resn_6, 17, num_classes)
        conv_18 = conv2d(conv_17, 18, num_classes)
        resn_7 = resnet(resn_6, conv_18, 7)

        conv_19 = conv2d(resn_7, 19, num_classes)
        conv_20 = conv2d(conv_19, 20, num_classes)
        resn_8 = resnet(resn_7, conv_20, 8)

        conv_21 = conv2d(resn_8, 21, num_classes)
        conv_22 = conv2d(conv_21, 22, num_classes)
        resn_9 = resnet(resn_8, conv_22, 9)

        conv_23 = conv2d(resn_9, 23, num_classes)
        conv_24 = conv2d(conv_23, 24, num_classes)
        resn_10 = resnet(resn_9, conv_24, 10)

        conv_25 = conv2d(resn_10, 25, num_classes)
        conv_26 = conv2d(conv_25, 26, num_classes)
        resn_11 = resnet(resn_10, conv_26, 11)
        conv_27 = conv2d(resn_11, 27, num_classes, stride=2)

        conv_28 = conv2d(conv_27, 28, num_classes)
        conv_29 = conv2d(conv_28, 29, num_classes)
        resn_12 = resnet(conv_27, conv_29, 12)

        conv_30 = conv2d(resn_12, 30, num_classes)
        conv_31 = conv2d(conv_30, 31, num_classes)
        resn_13 = resnet(resn_12, conv_31, 13)

        conv_32 = conv2d(resn_13, 32, num_classes)
        conv_33 = conv2d(conv_32, 33, num_classes)
        resn_14 = resnet(resn_13, conv_33, 14)

        conv_34 = conv2d(resn_14, 34, num_classes)
        conv_35 = conv2d(conv_34, 35, num_classes)
        resn_15 = resnet(resn_14, conv_35, 15)

        conv_36 = conv2d(resn_15, 36, num_classes)
        conv_37 = conv2d(conv_36, 37, num_classes)
        resn_16 = resnet(resn_15, conv_37, 16)

        conv_38 = conv2d(resn_16, 38, num_classes)
        conv_39 = conv2d(conv_38, 39, num_classes)
        resn_17 = resnet(resn_16, conv_39, 17)

        conv_40 = conv2d(resn_17, 40, num_classes)
        conv_41 = conv2d(conv_40, 41, num_classes)
        resn_18 = resnet(resn_17, conv_41, 18)

        conv_42 = conv2d(resn_18, 42, num_classes)
        conv_43 = conv2d(conv_42, 43, num_classes)
        resn_19 = resnet(resn_18, conv_43, 19)
        conv_44 = conv2d(resn_19, 44, num_classes, stride=2)

        conv_45 = conv2d(conv_44, 45, num_classes)
        conv_46 = conv2d(conv_45, 46, num_classes)
        resn_20 = resnet(conv_44, conv_46, 20)

        conv_47 = conv2d(resn_20, 47, num_classes)
        conv_48 = conv2d(conv_47, 48, num_classes)
        resn_21 = resnet(resn_20, conv_48, 21)

        conv_49 = conv2d(resn_21, 49, num_classes)
        conv_50 = conv2d(conv_49, 50, num_classes)
        resn_22 = resnet(resn_21, conv_50, 22)

        conv_51 = conv2d(resn_22, 51, num_classes)
        conv_52 = conv2d(conv_51, 52, num_classes)
        resn_23 = resnet(resn_22, conv_52, 23)  # [None, 13,13,1024]
    with tf.name_scope('SCALE'):
        with tf.name_scope('scale_1'):
            conv_53 = conv2d(resn_23, 53, num_classes)
            conv_54 = conv2d(conv_53, 54, num_classes)
            conv_55 = conv2d(conv_54, 55, num_classes)  # [None,14,14,512]
            conv_56 = conv2d(conv_55, 56, num_classes)
            conv_57 = conv2d(conv_56, 57, num_classes)
            conv_58 = conv2d(conv_57, 58, num_classes)  # [None,13 ,13,1024]
            conv_59 = conv2d(conv_58, 59, num_classes, batch_norm_and_activation=False, trainable=True)
            # [yolo layer] 6,7,8 # 82  --->predict    scale:1, stride:32, detecting large objects => mask: 6,7,8
            # 13x13x255, 255=3*(80+1+4)
        with tf.name_scope('scale_2'):
            route_1 = route1(conv_57, name="route_1")
            conv_60 = conv2d(route_1, 60, num_classes)
            upsam_1 = upsample(conv_60, 2, name="upsample_1")
            route_2 = route2(upsam_1, resn_19, name="route_2")
            conv_61 = conv2d(route_2, 61, num_classes)
            conv_62 = conv2d(conv_61, 62, num_classes)
            conv_63 = conv2d(conv_62, 63, num_classes)
            conv_64 = conv2d(conv_63, 64, num_classes)
            conv_65 = conv2d(conv_64, 65, num_classes)
            conv_66 = conv2d(conv_65, 66, num_classes)
            conv_67 = conv2d(conv_66, 67, num_classes, batch_norm_and_activation=False, trainable=True)
            # [yolo layer] 3,4,5 # 94  --->predict   scale:2, stride:16, detecting medium objects => mask: 3,4,5
            # 26x26x255, 255=3*(80+1+4)
        with tf.name_scope('scale_3'):
            route_3 = route1(conv_65, name="route_3")
            conv_68 = conv2d(route_3, 68, num_classes)
            upsam_2 = upsample(conv_68, 2, name="upsample_2")
            route_4 = route2(upsam_2, resn_11, name="route_4")
            conv_69 = conv2d(route_4, 69, num_classes)
            conv_70 = conv2d(conv_69, 70, num_classes)
            conv_71 = conv2d(conv_70, 71, num_classes)
            conv_72 = conv2d(conv_71, 72, num_classes)
            conv_73 = conv2d(conv_72, 73, num_classes)
            conv_74 = conv2d(conv_73, 74, num_classes)
            conv_75 = conv2d(conv_74, 75, num_classes, batch_norm_and_activation=False, trainable=True)
            # [yolo layer] 0,1,2 # 106 --predict scale:3, stride:8, detecting the smaller objects => mask: 0,1,2
            # 52x52x255, 255=3*(80+1+4)
            # Bounding Box:  YOLOv2: 13x13x5
            #                YOLOv3: 13x13x3x3, 3 for each scale

    return conv_59, conv_67, conv_75


def conv2d(inputs, idx, num_classes, stride=1, batch_norm_and_activation=True, trainable=False):
    name_conv = 'conv_' + str(idx)
    name_w = 'weights' + str(idx)
    name_b = 'biases' + str(idx)
    name_mean = 'moving_mean' + str(idx)
    name_vari = 'moving_variance' + str(idx)
    name_beta = 'beta' + str(idx)
    name_gam = 'gamma' + str(idx)
    # tous = True
    tous = False
    # tous = self.ST
    with tf.variable_scope(name_conv):
        if trainable:
            if idx == 59:
                weights = tf.Variable(
                    np.random.normal(size=[1, 1, 1024, 3 * (num_classes + 1 + 4)], loc=0.0, scale=0.01),
                    trainable=True,
                    dtype=np.float32, name="weights")
            elif idx == 67:
                weights = tf.Variable(
                    np.random.normal(size=[1, 1, 512, 3 * (num_classes + 1 + 4)], loc=0.0, scale=0.01),
                    trainable=True,
                    dtype=np.float32, name="weights")
            else:
                weights = tf.Variable(
                    np.random.normal(size=[1, 1, 256, 3 * (num_classes + 1 + 4)], loc=0.0, scale=0.01),
                    trainable=True,
                    dtype=np.float32, name="weights")
        else:
            weights = tf.Variable(W(idx), trainable=tous, dtype=tf.float32, name="weights")
        tf.summary.histogram(name_w, weights)  # add summary

        if stride == 2:
            paddings = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
            inputs_pad = tf.pad(inputs, paddings, "CONSTANT")
            conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID',
                                name="nn_conv")
        else:
            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME', name="conv")

        if batch_norm_and_activation:  #
            # conv_1 ---> conv_75 EXCEPT conv_59, conv_67, conv_75
            with tf.variable_scope('BatchNorm'):
                variance_epsilon = tf.constant(0.0001, name="epsilon")  # small float number to avoid dividing by 0

                moving_mean, moving_variance, beta, gamma = B(idx)
                moving_mean = tf.Variable(moving_mean, trainable=tous, dtype=tf.float32, name="moving_mean")
                tf.summary.histogram(name_mean, moving_mean)  # add summary
                moving_variance = tf.Variable(moving_variance, trainable=tous, dtype=tf.float32,
                                              name="moving_variance")
                tf.summary.histogram(name_vari, moving_variance)  # add summary
                beta = tf.Variable(beta, trainable=tous, dtype=tf.float32, name="beta")
                tf.summary.histogram(name_beta, beta)  # add summary
                gamma = tf.Variable(gamma, trainable=tous, dtype=tf.float32, name="gamma")
                tf.summary.histogram(name_gam, gamma)  # add summary
                conv = tf.nn.batch_normalization(conv, moving_mean, moving_variance, beta, gamma,
                                                 variance_epsilon, name='BatchNorm')
                # conv = tf.nn.batch_normalization(conv, mean, var, beta, gamma,
                #                                  variance_epsilon, name='BatchNorm')
            with tf.name_scope('Activation'):
                alpha = tf.constant(0.1, name="alpha")  # Slope of the activation function at x < 0
                acti = tf.maximum(alpha * conv, conv)
            return acti

        else:
            # for conv_59, conv67, conv_75
            if trainable:
                # biases may be  init =0
                biases = tf.Variable(
                    np.random.normal(size=[3 * (num_classes + 1 + 4), ], loc=0.0, scale=0.01),
                    trainable=True,
                    dtype=np.float32, name="biases")
            else:
                biases = tf.Variable(B(idx), trainable=False, dtype=tf.float32, name="biases")
            tf.summary.histogram(name_b, biases)  # add summary
            conv = tf.add(conv, biases)
            return conv


def route1(inputs, name):
    # [route]-4
    with tf.name_scope(name):
        output = inputs
        return output


def route2(input1, input2, name):
    with tf.name_scope(name):
        output = tf.concat([input1, input2], -1, name='concatenate')  # input1:-1, input2: 61
        return output


def upsample(inputs, size, name):
    with tf.name_scope(name):
        w = tf.shape(inputs)[1]  # 416
        h = tf.shape(inputs)[2]  # 416
        output = tf.image.resize_nearest_neighbor(inputs, [size * w, size * h])
        return output


def resnet(a, b, idx):
    name_res = 'resn' + str(idx)
    with tf.name_scope(name_res):
        resn = a + b
        return resn
