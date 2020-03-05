#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:kaiyinzhou
@Tensorflow 2.0
All of the following Code was follow Google BERT!
"""

from __future__ import absolute_import, division, print_function
import math
import tensorflow as tf
from fennlp.tools import create_initializer, reshape_to_matrix


def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
    output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor


class MultiAttentionLayer(tf.keras.layers.Layer):
    """
    Performs multi-headed attention from `from_tensor` to `to_tensor`.
    """

    def __init__(self,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 batch_size=None,
                 from_seq_length=None,
                 to_seq_length=None,
                 name=None,
                 **kwargs):
        super(MultiAttentionLayer, self).__init__(name=name, **kwargs)
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.num_attention_heads = num_attention_heads
        self.value_act = value_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        # `query_layer` =[B*F, N*H]
        self._query_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.query_act,
            name="query",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        # `value_layer` = [B*T, N*H]
        self._key_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.key_act,
            name="key",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        # `query_layer` =[B*T, N*H]
        self._value_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.value_act,
            name="value",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        self.drop_out = tf.keras.layers.Dropout(self.attention_probs_dropout_prob)
        self.built = True

    def call(self, from_tensor, to_tensor=None, attention_mask=None, is_training=True):
        from_shape = from_tensor.shape.as_list()
        to_shape = to_tensor.shape.as_list()
        if len(from_shape) != len(to_shape):
            raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")
        if len(from_shape) == 3:
            self.batch_size = from_shape[0]
            self.from_seq_length = from_shape[1]
            self.to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (self.batch_size is None or self.from_seq_length is None or self.to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified."
                )
        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)
        query_layer = self._query_layer(from_tensor_2d)
        key_layer = self._key_layer(to_tensor_2d)
        value_layer = self._value_layer(to_tensor_2d)
        # [B,N,F,H]
        query_layer = transpose_for_scores(query_layer, self.batch_size, self.num_attention_heads,
                                           self.from_seq_length, self.size_per_head)
        # [B,N,T,H]
        key_layer = transpose_for_scores(key_layer, self.batch_size, self.num_attention_heads,
                                         self.to_seq_length, self.size_per_head)
        # attention_score = [B,N,F,T]
        attention_score = tf.linalg.matmul(query_layer, key_layer, transpose_b=True)
        attention_score = tf.math.multiply(attention_score, 1.0 / math.sqrt(float(self.size_per_head)))
        if attention_mask is not None:
            # [batch_size, from_seq_length, to_seq_length]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            # Here we convert aim position to zero, masked position to -10000
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            # This could let score for masked very low
            attention_score += adder
        # sofmax
        attention_probs = tf.math.softmax(attention_score)
        attention_probs = self.drop_out(attention_probs, training=is_training)
        # value [B,N,T,H]
        value_layer = transpose_for_scores(value_layer, self.batch_size, self.num_attention_heads,
                                           self.to_seq_length, self.size_per_head)
        # context_layer [B,N,F,H]
        context_layer = tf.linalg.matmul(attention_probs, value_layer)
        # [B,F,N,H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        if self.do_return_2d_tensor:
            context_layer = tf.reshape(context_layer, [self.batch_size * self.from_seq_length,
                                                       self.num_attention_heads * self.size_per_head])
        else:
            context_layer = tf.reshape(context_layer, [self.batch_size, self.from_seq_length,
                                                       self.num_attention_heads * self.size_per_head])
        return context_layer


# Bahdanau2015
class BahdanauAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size,
                 v_initializer=None,
                 w_initializer=None,
                 u_initializer=None,
                 use_bias=True,
                 b_initializer='zero',
                 name='att', **kwargs):
        super(BahdanauAttentionLayer, self).__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.v_initializer = tf.keras.initializers.get(v_initializer)
        self.w_initializer = tf.keras.initializers.get(w_initializer)
        self.u_initializer = tf.keras.initializers.get(u_initializer)
        self.b_initializer = tf.keras.initializers.get(b_initializer)

    def build(self, input_shape):
        inner_size = input_shape.as_list()
        self.V = self.add_weight(
            name="V",
            shape=[self.hidden_size],
            initializer=self.v_initializer,
        )
        self.W = self.add_weight(
            name="W",
            shape=[inner_size[-1], self.hidden_size],
            initializer=self.w_initializer,
        )
        self.U = self.add_weight(
            name="U",
            shape=[inner_size[-1], self.hidden_size],
            initializer=self.u_initializer,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.hidden_size],
                initializer=self.b_initializer,
            )

        self.attn = tf.keras.layers.Dense(self.hidden_size, activation="tanh")

    def call(self, hidden_state, encoder_outputs,training=False):
        """
        :param hidden_state: [B,D]
        :param encoder_outputs: [B,T,D]
        :return:
        """
        hidden_inner_dim = hidden_state.get_shape().as_list()[-1]
        encoder_inner_dim = encoder_outputs.get_shape().as_list()[-1]
        if hidden_inner_dim != encoder_inner_dim:
            raise ValueError("The last shape of hidden_state and encoder_outputs must equal!")
        dens1 = tf.tensordot(tf.expand_dims(hidden_state, 1), self.W, axes=(2, 1))
        dens2 = tf.tensordot(encoder_outputs, self.U, axes=(2, 1))
        tanh_ = self.attn(dens1 + dens2 + self.bias) if self.use_bias else self.attn(dens1 + dens2)
        cij = tf.tensordot(tanh_, self.V, axes=(2, 0))
        alphas = tf.keras.backend.softmax(cij)
        output = tf.math.reduce_sum(encoder_outputs * tf.expand_dims(alphas, -1), axis=1)  # [B*T*2H]
        return output, alphas

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            "use_bias": self.use_bias,
            'v_initializer': tf.keras.initializers.serialize(self.v_initializer),
            'w_initializer': tf.keras.initializers.serialize(self.w_initializer),
            'u_initializer': tf.keras.initializers.serialize(self.u_initializer),
            'b_initializer': tf.keras.initializers.serialize(self.b_initializer),
        }
        base_config = super(BahdanauAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

