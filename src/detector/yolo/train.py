import os
import shutil
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tqdm
from PIL import ImageDraw, ImageFont, Image
from PIL import ImageFile

from detector.yolo import config as cfg
from detector.yolo.config import Input_shape, channels
from detector.yolo.detect_function import predict
from detector.yolo.loss_function import compute_loss
from detector.yolo.network_function import construct_model
from detector.yolo.utils.yolo_utils import get_training_data, read_anchors, read_classes, get_data_length, \
    letterbox_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(101)
tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("classes_path", "", "Path to txt file with classes")
tf.flags.DEFINE_string("yolo_anchors", "", "Path to txt file with yolo anchors")
tf.flags.DEFINE_string("train", "", "Path to txt file with train data")
tf.flags.DEFINE_string("valid", "", "Path to txt with valid data")
tf.flags.DEFINE_string("output", "", "Output directory")
tf.flags.DEFINE_string("train_npy", "", "Path to npy to be loaded")
tf.flags.DEFINE_string("test_npy", "", "Path to npy to be loaded")
FLAGS = tf.flags.FLAGS

if os.path.exists(FLAGS.output):
    shutil.rmtree(FLAGS.output)
os.makedirs(FLAGS.output)

classes_paths = FLAGS.classes_path
classes_data = read_classes(classes_paths)
anchors_paths = FLAGS.yolo_anchors
anchors = read_anchors(anchors_paths)

annotation_path_train = FLAGS.train
annotation_path_valid = FLAGS.valid

input_shape = (Input_shape, Input_shape)  # multiple of 32
tf.logging.info("Starting 1st session...")
num_image_train = get_data_length(annotation_path_train)
num_image_valid = get_data_length(annotation_path_valid)

tf.logging.info("Train samples: %d, Valid samples: %d" % (num_image_train, num_image_valid))


def detect_image(image, sess, boxes, scores, classes, input_placeholder, image_shape_placeholder):
    image = Image.fromarray(image)
    new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
    boxed_image, image_shape = letterbox_image(image, new_image_size)
    # boxed_image, image_shape = resize_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    image_data /= 255.
    inputs = np.expand_dims(image_data, 0)  # Add batch dimension. #

    out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
                                                  feed_dict={input_placeholder: inputs,
                                                             image_shape_placeholder: image_shape,
                                                             # self.is_training: False
                                                             })

    font = ImageFont.truetype(font=cfg.font_file,
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype(np.int32))
    thickness = (image.size[0] + image.size[1]) // 500  # do day cua BB

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = "sign"
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box  # y_min, x_min, y_max, x_max
        top = max(0, np.floor(top + 0.5).astype(np.int32))
        left = max(0, np.floor(left + 0.5).astype(np.int32))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype(np.int32))
        right = min(image.size[0], np.floor(right + 0.5).astype(np.int32))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for j in range(thickness):
            draw.rectangle([left + j, top + j, right - j, bottom - j], outline=(0, 255, 0))
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0, 255, 0))
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return np.asarray(image)


# Explicitly create a Graph object
def main():
    X = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, channels], name='Input')  # for image_data
    visualise_image_placeholder = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, 3],
                                                 name="Vis")
    is_training = tf.placeholder(tf.bool)
    with tf.name_scope("Target"):
        Y1 = tf.placeholder(tf.float32, shape=[None, Input_shape / 32, Input_shape / 32, 3, (5 + len(classes_data))],
                            name='target_S1')
        Y2 = tf.placeholder(tf.float32, shape=[None, Input_shape / 16, Input_shape / 16, 3, (5 + len(classes_data))],
                            name='target_S2')
        Y3 = tf.placeholder(tf.float32, shape=[None, Input_shape / 8, Input_shape / 8, 3, (5 + len(classes_data))],
                            name='target_S3')

    with tf.name_scope("visualisation"):
        detections_summary_op = tf.summary.image("detections", visualise_image_placeholder, collections=[])
    scale1, scale2, scale3 = construct_model(X, len(classes_data))
    scale_total = [scale1, scale2, scale3]
    image_shape_placeholder = tf.placeholder(tf.float32, shape=[2, ])
    boxes, scores, classes = predict(scale_total, anchors, len(classes_data), image_shape_placeholder,
                                     score_threshold=0.5, iou_threshold=cfg.ignore_thresh)

    y_predict = [Y1, Y2, Y3]
    loss = compute_loss(scale_total, y_predict, anchors, len(classes_data), print_loss=False)
    with tf.name_scope("Optimizer"):
        tf.summary.scalar("global step", tf.train.get_or_create_global_step())
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss,
                                                                        global_step=tf.train.get_or_create_global_step())
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    summary_op = tf.summary.merge_all()
    with tf.Session(config=config) as sess:
        # Merges all summaries collected in the default graph
        # Summary Writers
        train_summary_writer = tf.summary.FileWriter(FLAGS.output)
        validation_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.output, "eval"))
        sess.run(tf.global_variables_initializer())
        epochs = 100  #
        batch_size = 32  # consider
        best_loss_valid = 10e6
        current_step = 0
        for epoch in range(epochs):
            train_generator = get_training_data(annotation_path_train,
                                                input_shape, anchors,
                                                num_classes=len(classes_data),
                                                max_boxes=20,
                                                batch_size=batch_size)
            tf.logging.info("Epoch: {}/{}".format(epoch + 1, epochs))
            start_time = time.time()
            mean_loss_train = []
            with tqdm.tqdm(total=num_image_train) as pbar:
                for x_train, box_data_train, image_shape_train, y_train in train_generator:
                    summary_train, loss_train, _ = sess.run([summary_op, loss, optimizer],
                                                            feed_dict={X: (x_train / 255.),
                                                                       Y1: y_train[0],
                                                                       Y2: y_train[1],
                                                                       Y3: y_train[2],
                                                                       is_training: True})  # , options=run_options)
                    train_summary_writer.add_summary(summary_train, current_step)
                    # Flushes the event file to disk
                    mean_loss_train.append(loss_train)
                    pbar.update(len(x_train))
                    detected_images = []
                    for image in x_train[:3]:
                        detected_images.append(detect_image(image, sess, boxes, scores, classes, X, image_shape_placeholder))
                    vis_summary = sess.run(detections_summary_op, feed_dict={
                        visualise_image_placeholder: detected_images
                    })
                    train_summary_writer.add_summary(vis_summary, current_step)
                    pbar.set_postfix(OrderedDict({
                        "loss": "%.4f" % np.mean(mean_loss_train)
                    }))
                    current_step += 1

            mean_loss_train = np.mean(mean_loss_train)
            duration = time.time() - start_time

            mean_loss_valid = []
            valid_generator = get_training_data(annotation_path_valid,
                                                input_shape, anchors,
                                                num_classes=len(classes_data),
                                                max_boxes=20,
                                                batch_size=batch_size)

            for x_valid, box_data_valid, image_shape_valid, y_valid, in valid_generator:
                # Run summaries and measure accuracy on validation set
                summary_valid, loss_valid = sess.run([summary_op, loss],
                                                     feed_dict={X: (x_valid / 255.),
                                                                Y1: y_valid[0],
                                                                Y2: y_valid[1],
                                                                Y3: y_valid[2],
                                                                is_training: False})  # ,options=run_options)
                validation_summary_writer.add_summary(summary_valid, current_step)
                mean_loss_valid.append(loss_valid)
            mean_loss_valid = np.mean(mean_loss_valid)
            tf.logging.info("epoch %s / %s \ttrain_loss: %s,\tvalid_loss: %s" % (
                epoch + 1, epochs, mean_loss_train, mean_loss_valid))

            if best_loss_valid > mean_loss_valid:
                best_loss_valid = mean_loss_valid
                create_new_folder = os.path.join(FLAGS.output, "checkpoints")
                try:
                    os.mkdir(create_new_folder)
                except OSError:
                    pass
                checkpoint_path = os.path.join(create_new_folder, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=epoch)
                tf.logging.info("Model saved in file: %s" % checkpoint_path)

        tf.logging.info("Tuning completed!")
        train_summary_writer.close()
        validation_summary_writer.close()


if __name__ == '__main__':
    main()
