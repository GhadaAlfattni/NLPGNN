#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

import tensorflow as tf

from fennlp.gnn.messagepassing import MessagePassing
from fennlp.gnn.utils import *


class GraphSAGEConvolution(MessagePassing):
    def __init__(self,
                 out_features,
                 aggr='sum',
                 use_bias=True,
                 concat=True,
                 normalize=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(GraphSAGEConvolution, self).__init__(aggr=aggr, **kwargs)
        self.out_features = out_features
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.concat = concat
        self.normalize = normalize

    def build(self, input_shapes):
        node_embedding_shapes = input_shapes.node_embeddings
        # adjacency_list_shapes = input_shapes.adjacency_lists
        in_features = node_embedding_shapes[-1]
        if self.concat:
            in_features = 2 * in_features
        self.weight = self.add_weight(
            shape=(in_features, self.out_features),
            initializer=self.kernel_initializer,
            name='w',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_features,),
                initializer=self.bias_initializer,
                name='b',
            )
        self.built = True

    def message_function(self, edge_source_states, edge_source,  # x_j source
                         edge_target_states, edge_target,  # x_i target
                         num_incoming_to_node_per_message,  # degree target
                         num_outing_to_node_per_message,  # degree source
                         edge_type_idx, training):
        """
        :param edge_source_states: [M,H]
        :param edge_target_states: [M,H]
        :param num_incoming_to_node_per_message:[M]
        :param edge_type_idx:
        :param training:
        :return:
        """
        messages = edge_source_states
        return messages

    def call(self, inputs, training):
        adjacency_lists = inputs.adjacency_lists
        node_embeddings = inputs.node_embeddings
        if not self.concat:
            adjacency_lists = add_remain_self_loop(adjacency_lists, len(node_embeddings))
        aggr_out = self.propagate(GNNInput(node_embeddings, adjacency_lists), training)
        if self.concat:
            aggr_out = tf.concat([node_embeddings, aggr_out], -1)

        aggr_out = tf.linalg.matmul(aggr_out, self.weight)

        if self.use_bias:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = tf.math.l2_normalize(aggr_out, -1)
        return aggr_out
