#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
from fennlp.layers.embedding import *
class TextCNNModel(tf.keras.layers.Layer):
    def __init__(self,
                 config=None,
                 is_training=True,
                 use_one_hot_embeddings=True,
                 do_return_all_layers=True,
                 num_hidden_layers=12,
                 **kwargs):
        self.config = config
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.do_return_all_layers = do_return_all_layers
        self.num_hidden_layers = num_hidden_layers

        super(TextCNNModel, self).__init__(**kwargs)
    def build(self,input_shape):
        config = self.config
        # 这里默认随机初始化权重
        #TODO 载入词向量
        self.token_embedding = WDEmbedding(vocab_size=config.vocab_size,
                                           embedding_size=config.hidden_size,
                                           initializer_range=config.initializer_range,
                                           word_embedding_name="word_embeddings",
                                           use_one_hot_embedding=self.use_one_hot_embeddings,
                                           name="embeddings")

        self.segposembedding = SegPosEmbedding(use_token_type=True,
                                               is_training=self.is_training,
                                               hidden_dropout_prob=config.hidden_dropout_prob,
                                               token_type_vocab_size=config.type_vocab_size,
                                               token_type_embedding_name="token_type_embeddings",
                                               use_position_embeddings=True,
                                               position_embedding_name="position_embeddings",
                                               initializer_range=config.initializer_range,
                                               max_position_embeddings=config.max_position_embeddings,
                                               name="embeddings"
                                               )
