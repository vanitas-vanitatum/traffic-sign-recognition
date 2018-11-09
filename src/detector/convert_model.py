import argparse

import tensorflow as tf

from detector.model import model_fn


def serving_input_receiver():
    serialised_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None, None, 3],
                                      name="input_images")
    receiver_tensor = {"image": serialised_input}
    receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features=receiver_tensor, default_batch_size=None
    )
    return receiver


def create_model(model_path: str) -> tf.estimator.Estimator:
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   params={
                                       "text_scale": 512,
                                       "learning_rate": 0.001,
                                   },
                                   config=tf.estimator.RunConfig(
                                       log_step_count_steps=20,
                                       save_checkpoints_steps=20,
                                       save_summary_steps=20,
                                   ),
                                   model_dir=model_path)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_folder", help="Model dir")
    args = parser.parse_args()
    m = create_model(args.model_folder)
    m.export_saved_model(args.model_folder,
                         serving_input_receiver())
