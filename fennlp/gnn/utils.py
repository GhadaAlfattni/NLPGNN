#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
from typing import NamedTuple, List
import numpy as np
import tensorflow as tf
from scipy import sparse
import scipy.sparse as sp


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

def remove_self_loop(adjacency_lists):
    row = adjacency_lists[:, 0]
    col = adjacency_lists[:, 1]
    mask = row != col
    adjacency_lists = adjacency_lists[mask]

    return adjacency_lists


def maybe_num_nodes(index, num_nodes):
    return tf.reduce_max(index) + 1 if num_nodes is None else num_nodes


def masksoftmax(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index, num_nodes)
    inter = tf.math.unsorted_segment_max(data=src,
                                         segment_ids=index,
                                         num_segments=num_nodes)
    # out = src - tf.gather(inter, index)# 每一个维度减去最大的特征
    out = src
    out = tf.math.exp(out)
    inter = tf.math.unsorted_segment_sum(data=out, segment_ids=index, num_segments=num_nodes)
    out = out / (tf.gather(inter, index) + 1e-16)
    return out
