#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf
from fennlp.gnn.utils import *
from fennlp.gnn.GSConv import GraphSAGEConvolution


class GraphSAGE(tf.keras.Model):
    def __init__(self, dim, num_class, drop_rate, **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)
        self.conv1 = GraphSAGEConvolution(dim, aggr="sum")
        self.conv2 = GraphSAGEConvolution(dim, aggr="sum")

        self.dense1 = tf.keras.layers.Dense(dim)
        self.dense2 = tf.keras.layers.Dense(num_class)

        self.drop = tf.keras.layers.Dropout(drop_rate)

    def call(self, node_embeddings, edge_index, batch, training=True):
        edge_index = [edge_index]
        x = tf.nn.relu(self.conv1(GNNInput(node_embeddings, edge_index), training))
        x = tf.nn.relu(self.conv2(GNNInput(x, edge_index), training))
        x = batch_read_out(x, batch)
        x = tf.nn.relu(self.dense1(x))
        x = self.drop(x, training=training)
        x = self.dense2(x)
        return tf.math.softmax(x, -1)

    def predict(self, node_embeddings, edge_index, batch, training=False):
        return self(node_embeddings, edge_index, batch, training)
