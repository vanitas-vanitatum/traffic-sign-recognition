import numpy as np
from PIL import Image


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def get_data_length(annotation_path):
    with open(annotation_path) as f:
        return len(f.readlines())


def get_training_data(annotation_path, input_shape, anchors, num_classes, max_boxes=100, batch_size=32):
    image_batch = []
    box_batch = []
    image_shape_batch = []
    with open(annotation_path) as f:
        GG = f.readlines()
        np.random.shuffle(GG)
        for line_num, line in enumerate(GG):
            line = line.split(' ')
            filename = line[0]
            if filename[-1] == '\n':
                filename = filename[:-1]
            image = Image.open(filename)
            # For the case 2
            boxed_image, shape_image = letterbox_image(image, tuple(reversed(input_shape)))
            image_batch.append(np.array(boxed_image, dtype=np.uint8))  # pixel: [0:255] uint8:[-128, 127]
            image_shape_batch.append(np.array(shape_image))
            boxes = np.zeros((max_boxes, 5), dtype=np.int32)
            # correct the BBs to the image resize
            for i, box in enumerate(line[1:]):
                if i < max_boxes:
                    boxes[i] = np.array(list(map(int, box.split(','))))
                else:
                    break
                image_size = np.array(image.size)
                input_size = np.array(input_shape[::-1])
                # for case 2
                new_size = (image_size * np.min(input_size / image_size)).astype(np.int32)
                # Correct BB to new image
                boxes[i:i + 1, 0:2] = (
                        boxes[i:i + 1, 0:2] * new_size / image_size + (input_size - new_size) / 2).astype(np.int32)
                boxes[i:i + 1, 2:4] = (
                        boxes[i:i + 1, 2:4] * new_size / image_size + (input_size - new_size) / 2).astype(np.int32)
            box_batch.append(boxes)
            if len(image_batch) >= batch_size or line_num == len(GG):
                image_shape = np.array(image_shape_batch)
                image_data = np.array(image_batch)
                box_data = (np.array(box_batch))
                y_true = preprocess_true_boxes(box_data, input_shape[0], anchors, num_classes)
                yield image_data, box_data, image_shape, y_true

                image_shape_batch = []
                image_batch = []
                box_batch = []


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding
    :param: size: input_shape
    :return:boxed_image:
            image_shape: original shape (h, w)
    """
    image_w, image_h = image.size
    image_shape = np.array([image_h, image_w])
    w, h = size
    new_w = int(image_w * min(w / image_w, h / image_h))
    new_h = int(image_h * min(w / image_w, h / image_h))
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    return boxed_image, image_shape


def preprocess_true_boxes(true_boxes, Input_shape, anchors, num_classes):
    """
    Preprocess true boxes to training input format
    :param true_boxes: array, shape=(N, 100, 5)N:so luong anh,100:so object max trong 1 anh, 5:x_min,y_min,x_max,y_max,class_id
                    Absolute x_min, y_min, x_max, y_max, class_code reletive to input_shape.
    :param input_shape: array-like, hw, multiples of 32, shape = (2,)
    :param anchors: array, shape=(9, 2), wh
    :param num_classes: integer
    :return: y_true: list(3 array), shape like yolo_outputs, xywh are reletive value 3 array [N,, 13, 13, 3, 85]
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    true_boxes = np.array(true_boxes, dtype=np.float32)
    input_shape = np.array([Input_shape, Input_shape], dtype=np.int32)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # [m, T, 2]  (x, y)center point of BB
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # w = x_max - x_min  [m, T, 2]
    # h = y_max - y_min
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # hw -> wh
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]  # hw -> wh

    N = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(3)]
    # grid_shapes = [np.array(input_shape // scale, dtype=np.int) for scale in [32, 16, 8]]  # [2,] ---> [3, 2]
    y_true = [np.zeros((N, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + int(num_classes)),
                       dtype=np.float32) for l in range(3)]  # (m, 13, 13, 3, 85)

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # [1, 3, 2]
    anchor_maxes = anchors / 2.  # w/2, h/2  [1, 3, 2]
    anchor_mins = -anchor_maxes  # -w/2, -h/2  [1, 3, 2]
    valid_mask = boxes_wh[..., 0] > 0  # w>0 True, w=0 False

    for b in (range(N)):  # for all of N image
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  # image 0: wh [[[163., 144.]]]
        # Expand dim to apply broadcasting.
        if len(wh) == 0:
            continue
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for l in range(3):  # 1 in 3 scale
                if n in anchor_mask[l]:  # choose the corresponding mask: best_anchor in [6, 7, 8]or[3, 4, 5]or[0, 1, 2]

                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype(
                        np.int32)  # ex: 3+1.2=4.2--> vao ô co y=4
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype(
                        np.int32)  # ex: 3+0.5=3.5--> vao o co x=3 --> o (x,y)=(3,4)  # TODO
                    if grid_shapes[l][1] == 13 and (i >= 13 or j >= 13):
                        print(i)
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype(np.int32)  # idx classes in voc classes
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1  # score = 1
                    y_true[l][b, j, i, k, 5 + c] = 1  # classes = 1, the others =0
                    break

    return y_true
