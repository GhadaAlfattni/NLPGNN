#! encoding="utf-8"
import tensorflow as tf


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConvolution, self).__init__(self, **kwargs)
        
        self.kernel_initializer = tf.keras.initializers.get()
        bias_initializer
        kernel_regularizer
        bias_regularizer
        activity_regularizer
        kernel_constraint
        bias_constraint
