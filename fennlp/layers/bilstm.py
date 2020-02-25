#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from fennlp.datas.word2vec import glove

class BiLSTM(tf.keras.layers.Layer):
    def __init__(self,config,**kwargs):
        super(BiLSTM,self).__init__(**kwargs)
        self.config = config
        if config.init_weights:
            weights = glove(vocab_index_path=config.vocab_index_path, emb_dim=config.em_dim)
            self.embed = tf.keras.layers.Embedding(config.vocab_size, config.em_dim,
                                                   embeddings_initializer=tf.keras.initializers.constant(weights),
                                                   mask_zero=True)
        else:
            self.embed = tf.keras.layers.Embedding(config.vocab_size, config.em_dim)
        if config.startic_embedding:
            self.embed.trainable = False
        # self.bilstm  = tf.keras.layers.LSTM(config.lstm_dim,return_sequences=config.return_sequence)
        self.bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.lstm_dim,
                                                                         return_sequences=config.return_sequence,
                                                                         dropout=config.dropout_rate))
        self.dense = tf.keras.layers.Dense(config.output_dim,activation='softmax')
        # self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

    def call(self,input,training=True):
        embed = self.embed(input)
        embed = self.batchnorm1(embed)
        mask = self.embed.compute_mask(input)
        bilstm = self.bilstm1(embed,mask=mask)
        bilstm = self.batchnorm2(bilstm)
        logits = self.dense(bilstm)
        return logits






