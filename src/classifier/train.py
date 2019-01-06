from pathlib import Path

import numpy as np
import tensorflow as tf

from classifier.architectures.mobilenet2 import construct_model as mobilenet_construct_model
from classifier.data import DataLoader


def model_fn(features, labels, mode, params):
    num_classes = params["num_classes"]
    learning_rate = params.get("learning_rate", 0.001)

    labels = labels["label"]
    net = inputs = features["image"]

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    net = mobilenet_construct_model(net, is_training, num_classes)
    logits = tf.add(net, 0, name="output")
    y_pred = tf.nn.softmax(logits=logits)

    y_pred_cls = tf.argmax(y_pred, axis=1)
    predictions = {
        "class": y_pred_cls,
        "class_proba": y_pred,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels)
        loss = tf.reduce_mean(cross_entropy)

        total_loss = tf.add_n([loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        optimizer = tf.train.AdamOptimizer(learning_rate)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(
                loss=total_loss, global_step=tf.train.get_global_step()
            )

        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls),
            "precision": tf.metrics.precision(labels, y_pred_cls),
            "recall": tf.metrics.recall(labels, y_pred_cls),
            "false_positives": tf.metrics.false_positives(labels, y_pred_cls),
            "false_negatives": tf.metrics.false_negatives(labels, y_pred_cls),
        }
        for metric, op in metrics.items():
            tf.summary.scalar(f"{metric}_train", op[1])

        tf.summary.image("images", inputs, max_outputs=12)
        tf.summary.scalar("total_loss", loss)
        tf.summary.scalar("learning_rate", learning_rate)

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            predictions=predictions,
            eval_metric_ops=metrics
        )

    tf.logging.info("Model params: {}".format(np.prod(sum([
        np.prod(var.get_shape().as_list()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    ]))))

    return spec


def main(argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    folder_path = Path(FLAGS.model_folder)
    folder_path.mkdir(parents=True)

    patience_agg = 0
    best_loss = float("inf")

    train_loader = DataLoader(FLAGS.train_data, True)
    test_loader = DataLoader(FLAGS.test_data, False)

    train_loader.save_label_encoder((folder_path / "label_encoder.pkl"))
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   params={
                                       "learning_rate": FLAGS.learning_rate,
                                       "num_classes": train_loader.num_classes
                                   },
                                   model_dir=folder_path.as_posix())
    patience = FLAGS.patience

    for epoch in range(FLAGS.epochs):
        tf.logging.info(f"Epoch {epoch + 1}/{FLAGS.epochs}")

        model.train(input_fn=train_loader.input_fn(batch_size=FLAGS.batch_size), steps=2000)
        test_result = model.evaluate(input_fn=test_loader.input_fn(FLAGS.batch_size))

        if test_result["loss"] < best_loss:
            best_loss = test_result["loss"]
            patience_agg = 0
        else:
            patience_agg += 1
        if patience_agg == patience:
            tf.logging.info("Early stopping")
            break


if __name__ == '__main__':
    tf.flags.DEFINE_string('train_data', '', '')
    tf.flags.DEFINE_string('test_data', "", "")
    tf.flags.DEFINE_integer('batch_size', 64, '')
    tf.flags.DEFINE_integer('epochs', 1000, '')
    tf.flags.DEFINE_integer('patience', 10, '')
    tf.flags.DEFINE_string('model_folder', 'wololo', '')
    tf.flags.DEFINE_float('learning_rate', 0.001, '')
    FLAGS = tf.flags.FLAGS

    tf.app.run()
