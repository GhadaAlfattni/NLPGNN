#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
from typing import NamedTuple, List

import tensorflow as tf


class GNNInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: List


def add_remain_self_loop(adjacency_lists, num_nodes):
    loop_index = tf.range(0, num_nodes)
    loop_index = tf.expand_dims(loop_index, 1)
    loop_index = tf.tile(loop_index, [1, 2])
    row = adjacency_lists[:, 0]
    col = adjacency_lists[:, 1]
    mask = row != col
    loop_index = tf.concat([adjacency_lists[mask], loop_index], 0)
    return loop_index

def add_self_loop(adjacency_lists, num_nodes):
    loop_index = tf.range(0, num_nodes)
    loop_index = tf.expand_dims(loop_index, 1)
    loop_index = tf.tile(loop_index, [1, 2])
    loop_index = tf.concat([adjacency_lists, loop_index], 0)
    return loop_index

