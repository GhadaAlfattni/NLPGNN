#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf

def classfication_loss(sparse,from_logits=False,label_smoothing=0):
    if sparse:
        # loss_op = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,label_smoothing=label_smoothing)
        loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    else:
        loss_op = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)
    return loss_op



