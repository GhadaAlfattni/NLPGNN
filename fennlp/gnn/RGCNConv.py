#! encoding="utf-8"
import tensorflow as tf


class RelationalGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, output_dim,
                 num_relations,
                 num_bases,
                 activation='linear',
                 kernel_initializer=None,
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 **kwargs):
        super(RelationalGraphConvolution, self).__init__(self, **kwargs)

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.use_biase = use_bias
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

    def build(self, input_shape):
        in_features = input_shape[1]
        self.basis = self.add_weight(
            shape=(self.num_bases, in_features, self.out_features),
            name='basis',
        )

        self.att = self.add_weight(
            shape=(self.num_relations, self.num_bases),
            name='kernel',
        )
        # W_0^l
        self.weight = self.add_weight(
            shape=(in_features, self.out_features),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel',
        )
        # bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                name='bias',
                constraint=self.bias_constraint
            )

    def call(self, inputs):
        self.wr = tf.linalg.matmul(self.att, tf.reshape(self.basis, (self.num_bases, -1)))
        self.wr = tf.reshape(self.wr, (self.num_relations, -1, self.output_dim))



        if inputs is None:
            output = output + self.weight
        else:
            output = output + tf.linalg.matmul(inputs, self.weight)
        if self.use_biase:
            output += self.bias
        return output
