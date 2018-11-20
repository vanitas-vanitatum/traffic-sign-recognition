import tensorflow as tf


def spatial_dropout(inputs: tf.Tensor, drop_proba: float, is_training: bool) -> tf.Tensor:
    if not is_training:
        return inputs
    shape = tf.shape(inputs)
    keep_proba = 1 - drop_proba
    return tf.nn.dropout(inputs, keep_proba, noise_shape=[shape[0], 1, 1, shape[-1]])
