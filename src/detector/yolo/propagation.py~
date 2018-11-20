# from argparse import ArgumentParser
from PIL import Image, ImageFont, ImageDraw
from config import Input_shape, channels, threshold, ignore_thresh, path
from network_function import YOLOv3
from detect_function import predict
from utils.yolo_utils import read_anchors, read_classes, letterbox_image  # , resize_image
from timeit import default_timer as timer  # to calculate FPS
from pathlib import Path
import numpy as np
import tensorflow as tf
import argparse
import colorsys
import random
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

class YOLO(object):
    def __init__(self):

        self.anchors_path = path + '/yolo3/model/yolo_anchors.txt'
        self.COCO = False
        self.trainable = True

        args1 = sys.argv[2]
        if args1=='COCO':
            print("-----------COCO-----------")
            self.COCO = True
            self.classes_path = path + '/yolo3/model/coco_classes.txt'
            self.trainable = False
        elif args1=='VOC':
            print("-----------VOC------------")
            self.classes_path = path + '/yolo3/model/voc_classes.txt'
        elif args1=='boat':
            print("-----------boat-----------")
            self.classes_path = path + '/yolo3/model/boat_classes.txt'

        # args = self.argument()
        # if args.COCO:
        #     print("-----------COCO-----------")
        #     self.COCO = True
        #     self.classes_path = self.PATH + '/model/coco_classes.txt'
        #     self.trainable = False
        # elif args.VOC:
        #     print("-----------VOC------------")
        #     self.classes_path = self.PATH + '/model/voc_classes.txt'
        # elif args.boat:
        #     print("-----------boat-----------")
        #     self.classes_path = self.PATH + '/model/boat_classes.txt'

        self.class_names = read_classes(self.classes_path)
        self.anchors = read_anchors(self.anchors_path)
        self.threshold = 0.5# threshold
        self.ignore_thresh = ignore_thresh
        self.INPUT_SIZE = (Input_shape, Input_shape)  # fixed size or (None, None)
        self.is_fixed_size = self.INPUT_SIZE != (None, None)
        # LOADING SESSION...
        self.boxes, self.scores, self.classes, self.sess = self.load()

    @staticmethod
    def argument():
        parser = argparse.ArgumentParser(description='COCO or VOC or boat')
        parser.add_argument('--COCO', action='store_true', help='COCO flag')
        parser.add_argument('--VOC', action='store_true', help='VOC flag')
        parser.add_argument('--boat', action='store_true', help='boat flag')
        args = parser.parse_args()
        return args
    def load(self):
        # Remove nodes from graph or reset entire default graph
        tf.reset_default_graph()

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.x = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, channels])
        self.image_shape = tf.placeholder(tf.float32, shape=[2,])
        # self.is_training = tf.placeholder(tf.bool)
        # image_shape = np.array([image.size[0], image.size[1]])  # tf.placeholder(tf.float32, shape=[2,])

        # Generate output tensor targets for filtered bounding boxes.
        # scale1, scale2, scale3 = YOLOv3(self.x, len(self.class_names), trainable=self.trainable, is_training=self.is_training).feature_extractor()
        scale1, scale2, scale3 = YOLOv3(self.x, len(self.class_names), trainable=self.trainable).feature_extractor()
        scale_total = [scale1, scale2, scale3]

        # detect
        boxes, scores, classes = predict(scale_total, self.anchors, len(self.class_names), self.image_shape,
                                         score_threshold=self.threshold, iou_threshold=self.ignore_thresh)

        # Add ops to save and restore all the variables
        saver = tf.train.Saver(var_list=None if self.COCO==True else tf.trainable_variables())

        # Allowing GPU memory growth

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)
        sess.run(tf.global_variables_initializer())
        epoch = input('Entrer a check point at epoch:')
        # For the case of COCO
        epoch = epoch if self.COCO == False else 2000
        checkpoint = path + "/yolo3/save_model/SAVER_MODEL_boat10/model.ckpt-" + str(epoch)
        try:
            aaa = checkpoint + '.meta'
            my_abs_path = Path(aaa).resolve()
        except FileNotFoundError:
            print("Not yet training!")
        else:
            saver.restore(sess, checkpoint)
            print("checkpoint: ", checkpoint)
            print("already training!")

        return boxes, scores, classes, sess

    def detect_image(self, image):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        if self.is_fixed_size:
            assert self.INPUT_SIZE[0] % 32 == 0, 'Multiples of 32 required'
            assert self.INPUT_SIZE[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image, image_shape = letterbox_image(image, tuple(reversed(self.INPUT_SIZE)))
            # boxed_image, image_shape = resize_image(image, tuple(reversed(self.INPUT_SIZE)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image, image_shape = letterbox_image(image, new_image_size)
            # boxed_image, image_shape = resize_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print("heights, widths:", image_shape)
        image_data /= 255.
        inputs = np.expand_dims(image_data, 0)  # Add batch dimension. #

        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                           feed_dict={self.x: inputs,
                                                                      self.image_shape: image_shape,
                                                                      # self.is_training: False
                                                                      })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # Visualisation#################################################################################################
        font = ImageFont.truetype(font=path + '/yolo3/model/font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype(np.int32))
        thickness = (image.size[0] + image.size[1]) // 500  # do day cua BB

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
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
            print(label, (left, top), (right, bottom))  # (x_min, y_min), (x_max, y_max)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image

def detect_video(yolo, video_path=None, output_video=None):
    import urllib.request as urllib
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20.0  # display fps frame per second
    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    if video_path=='stream':
        url = 'http://10.18.97.1:8080/shot.jpg'
        out = cv2.VideoWriter(output_video, fourcc, fps, (1280, 720))
        while True:

            # Use urllib to get the image and convert into a cv2 usable format
            imgResp = urllib.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
            # print(np.shape(img))  # get w, h from here

            image = Image.fromarray(img)
            image = yolo.detect_image(image)
            result = np.asarray(image)

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", result)
            out.write(result)

            # To give the processor some less stress
            # time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        yolo.sess.close()
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        # The size of the frames to write
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        while True:
            ret, frame = cap.read()
            if ret==True:
                image = Image.fromarray(frame)

                image = yolo.detect_image(image)
                result = np.asarray(image)

                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.imshow("Result", result)

                out.write(result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        yolo.sess.close()

def detect_img(yolo, output=''):
    while True:
        img = input('Input image filename:')
        try:
            img = path + '/yolo3' + str(img)
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.save(output)
            r_image.show()

    yolo.sess.close()


if __name__ == '__main__':
    choose = sys.argv[1]
    if choose=='image':
        output = sys.argv[3]
        detect_img(YOLO(), output)
    elif choose=='video':
        video_path = sys.argv[3]
        output = sys.argv[4]
        detect_video(YOLO(), video_path, output)