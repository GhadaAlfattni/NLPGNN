#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf


class BiLSTM(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, vocab_size, embedding_size, dropout_rate, maxlen, **kwargs):
        super(BiLSTM, self).__init__(**kwargs)
        self.embed = tf.keras.layers.Embedding(vocab_size + 1, embedding_size, maxlen, mask_zero=True)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim,
                                 dropout=dropout_rate)
        )

    def call(self, inputs):
        embed = self.embed(inputs)
        logits = self.bilstm(embed)
        return logits
