#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import numpy as np
import tensorflow as tf
from fennlp.tools import get_shape_list


def einsum_via_matmul(input_tensor, w, num_inner_dims):
    input_shape = get_shape_list(input_tensor)
    w_shape = get_shape_list(w)
    batch_dims = input_shape[: -num_inner_dims]
    inner_dims = input_shape[-num_inner_dims:]
    outer_dims = w_shape[num_inner_dims:]
    inner_dim = np.prod(inner_dims)
    outer_dim = np.prod(outer_dims)
    if num_inner_dims > 1:
        input_tensor = tf.reshape(input_tensor, batch_dims + [inner_dim])
    if len(w_shape) > 2:
        w = tf.reshape(w, [inner_dim, outer_dim])
    ret = tf.matmul(input_tensor, w)
    if len(outer_dims) > 1:
        ret = tf.reshape(ret, batch_dims + outer_dims)
    return ret


class DenseLayer3d(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads,
                head_size, initializer, activation,
                use_einsum, name=None, **kwargs):
        super(DenseLayer3d, self).__init__(name=name, **kwargs)
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        self.initializer = initializer
        self.activation = activation
        self.use_einsum = use_einsum

    def build(self, input_shape):
        self.hidden_size = input_shape[2]
        self.w = self.add_weight(
            name="kernel",
            shape=[self.hidden_size, self.num_attention_heads * self.head_size],
            initializer=self.initializer,
            trainable=True
        )
        self.b = self.add_weight(
            name="bias",
            shape=[self.num_attention_heads * self.head_size],
            initializer=tf.zeros_initializer,
            trainable=True
        )
        self.built = True

    def call(self, input_tensor):
        w = tf.reshape(self.w, [self.hidden_size, self.num_attention_heads, self.head_size])
        b = tf.reshape(self.b, [self.num_attention_heads, self.head_size])
        if self.use_einsum:
            ret = tf.einsum("BFH,HND->BFND", input_tensor, w)
        else:
            ret = einsum_via_matmul(input_tensor, w, 1)
        ret += b
        if self.activation is not None:
            return self.activation(ret)
        else:
            return ret


class DenseLayer3dProj(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size,
                 head_size,
                 initializer,
                 activation,
                 use_einsum,
                 name=None,
                 **kwargs):

        super(DenseLayer3dProj, self).__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.initializer = initializer
        self.activation = activation
        self.use_einsum = use_einsum

    def build(self, input_shape):
        self.num_attention_heads = input_shape[2]
        self.w = self.add_weight(
            name="kernel",
            shape=[self.num_attention_heads * self.head_size, self.hidden_size],
            initializer=self.initializer)
        self.b = self.add_weight(
            name="bias", shape=[self.hidden_size], initializer=tf.zeros_initializer)
        self.built = True

    def call(self, input_tensor):
        w = tf.reshape(self.w, [self.num_attention_heads, self.head_size,
                                self.hidden_size])
        if self.use_einsum:
            ret = tf.einsum("BFND,NDH->BFH", input_tensor, w)
        else:
            ret = einsum_via_matmul(input_tensor, w, 2)
        ret += self.b
        if self.activation is not None:
            return self.activation(ret)
        else:
            return ret


class DenseLayer2d(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 initializer,
                 activation,
                 use_einsum,
                 num_attention_heads=1,
                 name=None,
                 **kwargs):
        super(DenseLayer2d, self).__init__(name=name, **kwargs)
        self.output_size = output_size
        self.initializer = initializer
        self.activation = activation
        self.use_einsum = use_einsum
        self.num_attention_heads = num_attention_heads

    def build(self, input_shape):
        hidden_size = input_shape[2]
        self.w = self.add_weight(
            name="kernel",
            shape=[hidden_size, self.output_size],
            initializer=self.initializer)
        self.b = self.add_weight(
            name="bias", shape=[self.output_size],
            initializer=tf.zeros_initializer)
        self.built = True

    def call(self, input_tensor):
        if self.use_einsum:
            ret = tf.einsum("BFH,HO->BFO", input_tensor, self.w)
        else:
            ret = tf.matmul(input_tensor, self.w)
        ret += self.b
        if self.activation is not None:
            return self.activation(ret)
        else:
            return ret
