import tensorflow as tf


class MaskSparseCategoricalCrossentropy():
    def __init__(self, from_logits, use_mask=False):
        self.from_logits = from_logits
        self.use_mask = use_mask

    def __call__(self, y_true, y_predict, input_mask=None):
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_predict, self.from_logits)
        if self.use_mask:
            input_mask = tf.cast(input_mask, dtype=tf.float32)
            input_mask /= tf.reduce_mean(input_mask)
            cross_entropy *= input_mask
            # mask loss
            return tf.reduce_mean(cross_entropy)
        else:
            return tf.reduce_mean(cross_entropy)


class MaskCategoricalCrossentropy():
    def __init__(self, from_logits=False, use_mask=True):
        self.from_logits = from_logits
        self.use_mask = use_mask

    def __call__(self, y_true, y_predict, input_mask=None):
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_predict, self.from_logits)
        if self.use_mask:
            input_mask = tf.cast(input_mask, dtype=tf.float32)
            input_mask /= tf.reduce_mean(input_mask)
            cross_entropy *= input_mask
            # mask loss
            return tf.reduce_mean(cross_entropy)
        else:
            return tf.reduce_mean(cross_entropy)
