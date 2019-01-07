import argparse
import os

import tensorflow as tf

from classifier.architectures.mobilenet2 import construct_model


def dump_model_to_correct_pbtxt(checkpoint_path: str, output_path: str, number_of_classes: int):
    if os.path.exists(output_path):
        raise ValueError("Given path already exists!: %s" % output_path)
    with tf.get_default_graph().as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        output = construct_model(inputs, False, number_of_classes)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            tf.train.write_graph(sess.graph_def, os.path.dirname(output_path), os.path.basename(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_folder", help="Model dir")
    parser.add_argument("-o", "--output_folder", help="Output directory")
    parser.add_argument("-n", "--num_classes", help="Number of classes", type=int, default=33)
    args = parser.parse_args()
    dump_model_to_correct_pbtxt(args.model_folder, args.output_folder, args.num_classes)
