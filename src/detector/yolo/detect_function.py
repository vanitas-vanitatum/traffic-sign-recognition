import tensorflow as tf

from detector.yolo.config import Input_shape, threshold, ignore_thresh


def yolo_head(feature_maps, anchors, num_classes, input_shape, calc_loss=False):
    """
    Convert final layer features to bounding box parameters.
    (Features learned by the convolutional layers ---> a classifier/regressor which makes the detection prediction)
    :param feature_maps: the feature maps learned by the convolutional layers
                         3 scale [None, 13, 13, 255] from yolov3 structure anchors:[116, 90], [156, 198], [373, 326]
                                 [None, 26, 26, 255]                               [30, 61], [62, 45], [59, 119]
                                 [None, 52, 52, 255]                               [10, 13], [16, 30], [33, 23]
    :param anchors: 3 anchors for each scale shape=(3,2)
    :param num_classes: 80 for COCO
    :param input_shape: 416,416
    :return: box_xy  [None, 13, 13, 3, 2], 2: x,y center point of BB
             box_wh  [None, 13, 13, 3, 2], 2: w,h
             box_conf  [None, 13, 13, 3, 1], 1: conf
             box_class_pred  [None, 13, 13, 3, 80], 80: prob of each class
    """
    num_anchors = len(anchors)  # 3
    # Reshape to batch, height, width, num_anchors, box_params
    anchors_tensor = tf.cast(anchors, dtype=feature_maps.dtype)
    anchors_tensor = tf.reshape(anchors_tensor, [1, 1, 1, num_anchors, 2])  # shape=[1,1,1,3,2]

    # CREATE A GRID FOR EACH SCALE
    with tf.name_scope('Create_GRID'):
        grid_shape = tf.shape(feature_maps)[1:3]  # height, width ---> grid 13x13 for scale1
        #         (0,0) (1,0) ...                                      grid 26x26 for scale2
        #         (0,1) (1,1) ...                                      grid 52x52 for scale3
        #          ...
        # In YOLO the height index is the inner most iteration.
        grid_y = tf.range(0, grid_shape[0])  # array([0,1,...,11,12])
        grid_x = tf.range(0, grid_shape[1])
        grid_y = tf.reshape(grid_y, [-1, 1, 1, 1])  # shape=([13,  1,  1,  1])
        grid_x = tf.reshape(grid_x, [1, -1, 1, 1])  # [1, 13, 1, 1]
        grid_y = tf.tile(grid_y, [1, grid_shape[1], 1, 1])  # [13, 1, 1, 1] ---> [13, 13, 1, 1]
        grid_x = tf.tile(grid_x, [grid_shape[0], 1, 1, 1])  # [1, 13, 1, 1] ---> [13, 13, 1, 1]
        grid = tf.concat([grid_x, grid_y], axis=-1)  # shape=[13, 13,  1,  2]
        grid = tf.cast(grid, dtype=feature_maps.dtype)  # change type

    # Reshape [None, 13, 13, 255] =>[None, 13, 13, 3, 85]
    feature_maps_reshape = tf.reshape(feature_maps, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    with tf.name_scope('top_feature_maps'):
        # top of feature maps is a activation function
        """softmax is used for the multi-class logistic regression: ouput fall into [-1,1] ---> sum(all classes) = 1
        # sigmoid for the the 2-class logistic regression: output fall into [0,1] ---> sum(all classes) >1
        # for the relative width and weight, use the exponential function"""
        box_xy = tf.sigmoid(feature_maps_reshape[..., :2], name='x_y')  # [None, 13, 13, 3, 2]
        tf.summary.histogram(box_xy.op.name + '/activations', box_xy)
        box_wh = tf.exp(feature_maps_reshape[..., 2:4], name='w_h')  # [None, 13, 13, 3, 2]
        tf.summary.histogram(box_wh.op.name + '/activations', box_wh)
        box_confidence = tf.sigmoid(feature_maps_reshape[..., 4:5], name='confidence')  # [None, 13, 13, 3, 1]
        tf.summary.histogram(box_confidence.op.name + '/activations', box_confidence)
        box_class_probs = tf.sigmoid(feature_maps_reshape[..., 5:], name='class_probs')  # [None, 13, 13, 3, 80]
        tf.summary.histogram(box_class_probs.op.name + '/activations', box_class_probs)
        # Adjust predictions to each spatial grid point and anchor size.
        # Note: YOLO iterates over height index before width index.
        box_xy = (box_xy + grid) / tf.cast(grid_shape[::-1],  # (x,y + grid)/13. ---> in between (0., 1.)
                                           dtype=feature_maps_reshape.dtype)  # [None, 13, 13, 3, 2]
        box_wh = box_wh * anchors_tensor / tf.cast(input_shape[::-1],  # following to the scale
                                                   dtype=feature_maps_reshape.dtype)  # [None, 13, 13, 3, 2]

    if calc_loss:
        return grid, feature_maps_reshape, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """
    Convert YOLO box predictions to bounding box corners.(get corrected boxes)
    :param box_xy: (None, 13, 13, 3, 2) is box_x,y output of yolo_head()
    :param box_wh: (None, 13, 13, 3, 2) is box_w,h output of yolo_head()
    :param input_shape: 416,416
    :param image_shape: shape of input image (normal: height, width) . tf.placeholder(shape=(2, ))
    :return: box(2 corners) in original image shape (BB corresponding to h and w of image)
                        : 1 (y_min,x_min) left bottom corner
                          1 (y_max,x_max) right top corner
                    ---> (..., (y_min,x_min,y_max,x_max)) (None, 13, 13, 3, 4)
    """
    # Note: YOLO iterates over height index before width index.
    # batch_size = 3 #tf.shape(image_shape)[0]
    box_yx = box_xy[..., ::-1]  # (None, 13, 13, 3, 2) => ex: , x,y --> y,x
    box_hw = box_wh[..., ::-1]  # (None, 13, 13, 3, 2) w,h--->h,w
    input_shape = tf.cast(input_shape, dtype=box_yx.dtype)  # ex: (416,416)
    image_shape = tf.cast(image_shape, dtype=box_yx.dtype)  # ex: (720, 1028)

    with tf.name_scope('resize_to_scale_correspond'):
        """un image (640, 480) to scale1 (stride 32)(13x13)
        ---> new shape = (13, 10)"""
        constant = (input_shape / image_shape)
        min = tf.minimum(constant[0], constant[1])
        new_shape = image_shape * min  # 640*(416/640), 480*(416/640)
        new_shape = tf.round(new_shape)  # lam tron ---> (416, 312)

    offset = (input_shape - new_shape) / (input_shape * 2.)  # 0,  (416-312)/2/416=0.125
    scale = input_shape / new_shape  # (1, 416/312)

    with tf.name_scope('return_corners_box'):
        # box in scale
        box_yx = (box_yx - offset) * scale  # (x-0)*1, (y-0.125)*416/312
        box_hw *= scale  # h*1, w*1.333
        box_mins = box_yx - (box_hw / 2.)  # (x-0)*1-h*1/2 = y_min, (y-0.125)*(416/312)-w*(416/312)/2 = x_min
        box_maxes = box_yx + (box_hw / 2.)  # (x-0)*1+h*1/2 = y_max, (y-0.125)*(416/312)+w*(416/312)/2 = x_max
        boxes = tf.concat([box_mins[..., 0:1],  # y_min
                           box_mins[..., 1:2],  # x_min
                           box_maxes[..., 0:1],  # y_max
                           box_maxes[..., 1:2]],  # x_max
                          axis=-1, name='box_in_scale')
        boxes = tf.multiply(boxes,
                            tf.concat([image_shape, image_shape], axis=-1), name='box_in_original_image_shape')
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    Process Conv layer output
    :param feats: [None, 13, 13, 255] is output of build_networks() (from network_function scripts)
    :param anchors: for yolo_head()
    :param num_classes: for yolo_head(), 80 for COCO
    :param input_shape: see yolo_head(), yolo_correct_boxes()
    :param image_shape: tensor targets for filtered bounding boxes. tf.placeholder(shape=(2, ))
    :return: boxes: [None*13*13*3, 4], predicted BBs with 4: cordoning of one BB (y_min,x_min, y_max,x_max)
             box_scores: [None*13*13*3, 80], 80: score= confidence * class_probability
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape,
                               image_shape)  # shape = (None, 13, 13, 3, 4), 4:(y_min,x_min,y_max,x_max)
    boxes = tf.reshape(boxes, [-1, 4], name='boxes')  # shape = (None*13*13*3, 4)

    with tf.name_scope('box_scores'):
        box_scores = box_confidence * box_class_probs  # (..., 1) * (..., 80) ---> (None, 13, 13, 3, 80)
        box_scores = tf.reshape(box_scores, [-1, num_classes])  # (None*13*13*3, 80)

    return boxes, box_scores


def predict(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=threshold,
            iou_threshold=ignore_thresh):
    """
    Evaluate YOLO model on given input and return filtered boxes
    :param yolo_outputs:
    :param anchors: [9,2]
    :param num_classes:
    :param image_shape: see yolo_boxes_and_scores()
    :param max_boxes: a scalar integer who present the maximum number of boxes to be selected by non max suppression
    :param score_threshold: score_threshold=.6
    :param iou_threshold: iou_threshold=.5
    :return:
    """
    # input_shape = tf.shape(yolo_outputs[0])[1:3] * 32  # scale1 13*32=416 [416,416]
    boxes = []
    box_scores = []
    input_shape = (Input_shape, Input_shape)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    for mask in range(3):  # 3 scale
        name = 'predict' + str(mask + 1)
        with tf.name_scope(name):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[mask],
                                                        anchors[anchor_mask[mask]],
                                                        num_classes,
                                                        input_shape,
                                                        image_shape)

            boxes.append(_boxes)  # list(3 array): [3, None*13*13*3, 4]
            box_scores.append(_box_scores)  # list(3 array): [3, None*13*13*3, 80]

    boxes = tf.concat(boxes, axis=0)  # [3 *None*13*13*3, 4]
    box_scores = tf.concat(box_scores, axis=0)  # [3 *None*13*13*3, 80]

    mask = box_scores >= score_threshold  # False & True in [3*None*13*13*3, 80] based on box_scores
    # maximum number of boxes to be selected by non max suppression
    max_boxes_tensor = tf.constant(max_boxes, dtype='int32', name='max_boxes')

    boxes_ = []
    scores_ = []
    classes_ = []

    for Class in range(num_classes):
        # name = 'Class'+str(Class)
        # with tf.name_scope(name):
        class_boxes = tf.boolean_mask(boxes, mask[:,
                                             Class])  # obj:[3 *None*13*13*3, 4], mask:[3 *None*13*13*3, 1] ---> [..., 4], each class: keep boxes who have (box_scores >= score_threshold)
        class_box_scores = tf.boolean_mask(box_scores[:, Class], mask[:, Class])  # [..., 1]

        nms_index = tf.image.non_max_suppression(class_boxes,  # [num_box(True), 4]
                                                 class_box_scores,  # [num_box(True), 1]
                                                 max_boxes_tensor,  # 20
                                                 iou_threshold=iou_threshold,
                                                 name='non_max_suppression')  # return an integer tensor of indices has the shape [M], M <= 20
        class_boxes = tf.gather(class_boxes,
                                nms_index,
                                name='TopLeft_BottomRight')  # Take the elements of indices (nms_index) in the class_boxes. [M, 4]
        class_box_scores = tf.gather(class_box_scores, nms_index, name='Box_score')  # [M, 1]
        with tf.name_scope('Class_prob'):
            classes = tf.ones_like(class_box_scores, 'int32') * Class  # [M, 1]
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = tf.concat(boxes_, axis=0, name='TopLeft_BottomRight')  # [N, 4] with N: number of objects
    scores_ = tf.concat(scores_, axis=0)  # [N,]
    classes_ = tf.concat(classes_, axis=0)  # [N,]

    return boxes_, scores_, classes_
