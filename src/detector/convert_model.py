import argparse
import os

import tensorflow as tf

from detector.model import model


def dump_model_to_correct_pbtxt(checkpoint_path: str, output_path: str):
    if os.path.exists(output_path):
        raise ValueError("Given path already exists!: %s" % output_path)
    with tf.get_default_graph().as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0),
                                      trainable=False)

        f_score, f_geometry = model(inputs, 512, is_training=True)
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
    args = parser.parse_args()
    dump_model_to_correct_pbtxt(args.model_folder, args.output_folder)
