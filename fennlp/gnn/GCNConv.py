#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

import tensorflow as tf

from fennlp.gnn.messagepassing import MessagePassing


class GraphConvolution(MessagePassing):
    def __init__(self,
                 out_features,
                 epsion=1e-7,
                 aggr="sum",
                 normalize=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 regularizer=None,
                 use_bias=False,
                 **kwargs):
        super(GraphConvolution, self).__init__(aggr, **kwargs)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.regularizer = tf.keras.initializers.get(regularizer)
        self.use_bias = use_bias
        self.normalize = normalize
        self.out_features = out_features
        self.epsion = epsion

    def build(self, input_shapes):
        node_embedding_shapes = input_shapes.node_embeddings
        adjacency_list_shapes = input_shapes.adjacency_lists
        num_edge_type = len(adjacency_list_shapes)
        in_features = node_embedding_shapes[-1]
        self._edge_type_weights = []
        self._edge_type_bias = []
        for i in range(num_edge_type):
            weight = self.add_weight(
                shape=(in_features, self.out_features),
                initializer=self.kernel_initializer,
                regularizer=self.regularizer,
                name='wt',
            )
            self._edge_type_weights.append(weight)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_features),
                initializer=self.bias_initializer,
                name='b',
            )
        else:
            self.bias = None
        self.cached_result = None

        self.built = True

    def message_function(self,
                         edge_source_states,  # x_j source
                         edge_target_states,  # x_i target
                         num_incoming_to_node_per_message,  # degree target
                         num_outing_to_node_per_message,  # degree source
                         edge_type_idx):
        """
        :param edge_source_states: [M,H]
        :param edge_target_states: [M,H]
        :param num_incoming_to_node_per_message:[M]
        :param edge_type_idx:
        :param training:
        :return:
        """
        # @lru_cache()
        # def normer():
        #     deg_target = tf.math.pow(num_incoming_to_node_per_message, -0.5)  # [M]
        #     deg_source = tf.math.pow(num_outing_to_node_per_message, -0.5)  # [M]
        #     norm = deg_source * deg_target
        #     return norm
        # norm = normer()

        weight_r = self._edge_type_weights[edge_type_idx]

        messages = tf.linalg.matmul(edge_source_states, weight_r)

        # deg_source = tf.where(tf.not_equal(deg_source, float('inf')), deg_source, tf.zeros_like(deg_source))
        # deg_target = tf.where(tf.not_equal(deg_target, float('inf')), deg_target, tf.zeros_like(deg_target))
        deg_target = tf.math.pow(num_incoming_to_node_per_message, -0.5)  # [M]
        deg_source = tf.math.pow(num_outing_to_node_per_message, -0.5)  # [M]
        norm = deg_source * deg_target
        messages = tf.reshape(norm, [-1, 1]) * messages
        return messages

    def call(self, inputs, training=True):
        aggr_out = self.propagate(inputs)
        if self.bias:
            aggr_out += + self.bias

        return aggr_out
