import shutil
from pathlib import Path

import tensorflow as tf

import common
from detector.icdar import train_input_fn, test_input_fn
from detector.model import model_fn


def main(argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    folder_path = Path(FLAGS.model_folder)
    if folder_path.exists():
        shutil.rmtree(folder_path.as_posix())
    folder_path.mkdir(parents=True)

    model = tf.estimator.Estimator(model_fn=model_fn,
                                   params={
                                       "text_scale": FLAGS.text_scale,
                                       "learning_rate": FLAGS.learning_rate,
                                   },
                                   model_dir=folder_path.as_posix())
    patience = FLAGS.patience
    patience_agg = 0
    best_loss = float("inf")
    for epoch in range(FLAGS.epochs):
        tf.logging.info(f"Epoch {epoch + 1}/{FLAGS.epochs}")
        model.train(input_fn=train_input_fn(common.DETECTOR_DATA_PATH, batch_size=FLAGS.batch_size))

        train_result = model.evaluate(input_fn=train_input_fn(common.DETECTOR_DATA_PATH, batch_size=FLAGS.batch_size))
        test_result = model.evaluate(input_fn=test_input_fn(common.DETECTOR_DATA_PATH, batch_size=FLAGS.batch_size))

        if test_result < best_loss:
            best_loss = test_result
            patience_agg = 0
        else:
            patience_agg += 1
        if patience_agg == patience:
            tf.logging.info("Early stopping")
            break


if __name__ == '__main__':
    tf.flags.DEFINE_integer('input_size', 512, '')
    tf.flags.DEFINE_integer('batch_size', 1, '')
    tf.flags.DEFINE_integer('text_scale', 512, '')
    tf.flags.DEFINE_integer('epochs', 1000, '')
    tf.flags.DEFINE_integer('patience', 10, '')
    tf.flags.DEFINE_string('model_folder', 'wololo', '')
    tf.flags.DEFINE_float('learning_rate', 0.0001, '')
    FLAGS = tf.flags.FLAGS

    tf.app.run()
