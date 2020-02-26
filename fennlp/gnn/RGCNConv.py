#! encoding="utf-8"
import tensorflow as tf


class RelationalGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_relations, num_bases):
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

    def build(self):
        pass


    def call(self):
        pass
