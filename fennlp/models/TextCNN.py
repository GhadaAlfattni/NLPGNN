#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf


class TextCNN(tf.keras.Model):
    def __init__(self,
                 maxlen,
                 vocab_size,
                 embedding_dims,
                 filter_size=[128, 128, 128],
                 kernel_sizes=[2, 3, 4],
                 class_num=2,
                 name=None,
                 **kwargs):
        super(TextCNN, self).__init__(name=name, **kwargs)
        self.maxlen = maxlen
        self.kernel_sizes = kernel_sizes

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dims, maxlen)
        self.convs = []
        self.max_poolings = []
        for i, k in enumerate(kernel_sizes):
            # Conv1D inputs=[None,maxlen,embedding_dim]
            self.convs.append(tf.keras.layers.Conv1D(filter_size[i],
                                                     k,
                                                     activation="relu"))
            # here a little different from methods provide in article,
            # but they have the same purpose
            self.max_poolings.append(tf.keras.layers.GlobalAvgPool1D())
        self.dense = tf.keras.layers.Dense(class_num, activation='softmax')

    def call(self, inputs):
        embedding = self.embedding(inputs)
        convs = []
        for i, k in enumerate(self.kernel_sizes):
            out = self.convs[i](embedding)
            out = self.max_poolings[i](out)
            convs.append(out)

        out = tf.keras.layers.concatenate(convs)
        out = self.dense(out)
        return out
