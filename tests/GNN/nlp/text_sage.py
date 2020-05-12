#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import time
import numpy as np
import tensorflow as tf
from fennlp.datas import Sminarog
from fennlp.metrics import Losess, Metric
from fennlp.models import GraphSAGE
from fennlp.gnn.utils import merge_batch_graph
from fennlp.callbacks import EarlyStopping

dim = 200
num_class = 52
drop_rate = 0.5
epoch = 100
penalty = 1e-4
lr = 1e-3

# R8,R52
data = Sminarog(data="R52", data_dir="data", embedding="glove200")
features, adjs, edge_attrs, labels, batchs, edge2index = data.build_graph(mode="train", p=3, k=5)
features1, adjs1, edge_attrs1, labels1, batchs1, _ = data.build_graph(edge2index, mode="test", p=3)


class TextGCNDynamicWeight(tf.keras.layers.Layer):
    def __init__(self, dim, num_class, drop_rate, **kwargs):
        super(TextGCNDynamicWeight, self).__init__(**kwargs)

        self.model = GraphSAGE(dim, num_class, drop_rate)

    # def build(self, input_shape):
    #     self.weight = self.add_weight(
    #         shape=(len(edge2index),),
    #         initializer='ones',
    #         name='an',
    #     )

    def call(self, feature, adj, edge_attr, batch, training=True):
        # edge_attr = tf.cast(edge_attr, dtype=tf.int32)
        # edge_weight = tf.gather(self.weight, edge_attr)

        # predict = self.model(feature, adj, batch, edge_weight, training=training)
        predict = self.model(feature, adj, batch, training=training)
        return predict

    def predict(self, feature, adj, edge_attr, batch, training=False):
        return self(feature, adj, edge_attr, batch, training)


accs_all = []
for i in range(10):
    model = TextGCNDynamicWeight(dim, num_class, drop_rate)
    optimize = tf.optimizers.Adam(lr)

    cross_entropy = Losess.MaskSparseCategoricalCrossentropy()
    acc_score = Metric.SparseAccuracy()

    stop_monitor = EarlyStopping(monitor="loss", patience=10,)
    for i in range(epoch):
        loss_train = []
        acc_train = []
        t = time.time()

        for feature, label, adj, edge_attr, batch in data.load(features[:-500], adjs[:-500],
                                                               labels[:-500], edge_attrs[:-500],
                                                               batchs[:-500], batch_size=32):
            feature, label, adj, edge_attr, batch = merge_batch_graph(feature, label, adj,
                                                                      edge_attr, batch)
            with tf.GradientTape() as tape:
                predict = model(feature, adj, edge_attr, batch, training=True)
                loss = cross_entropy(label, predict)
                loss += tf.add_n([tf.nn.l2_loss(v) for v in model.variables
                                  if "bias" not in v.name]) * penalty
                acc = acc_score(label, predict)
                loss_train.append(loss)
                acc_train.append(acc)

            grads = tape.gradient(loss, model.trainable_variables)
            optimize.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

        loss_valid = []
        acc_valid = []
        for feature, label, adj, edge_attr, batch in data.load(features[-500:], adjs[-500:],
                                                               labels[-500:], edge_attrs[-500:],
                                                               batchs[-500:],
                                                               batch_size=32):
            feature, label, adj, edge_attr, batch = merge_batch_graph(feature, label, adj,
                                                                      edge_attr, batch)
            t_predict = model.predict(feature, adj, edge_attr, batch, training=False)
            t_loss = cross_entropy(label, t_predict)
            t_acc = acc_score(label, t_predict)
            acc_valid.append(t_acc)
            loss_valid.append(t_loss.numpy())
        print("Valid: Epoch {} | Loss {:.4f} | Acc {:.4f} | Time {:.4f}".format(i, np.mean(loss_valid),
                                                                                np.mean(acc_valid),
                                                                                time.time() - t))
        # if stop_monitor(np.mean(loss_valid), model):
        #     break

    # test
    loss_test = []
    acc_test = []
    for feature, label, adj, edge_attr, batch in data.load(features1, adjs1,
                                                           labels1, edge_attrs1,
                                                           batchs1, batch_size=32):
        feature, label, adj, edge_attr, batch = merge_batch_graph(feature, label, adj,
                                                                  edge_attr, batch)
        t_predict = model.predict(feature, adj, edge_attr, batch, training=False)
        t_loss = cross_entropy(label, t_predict)
        t_acc = acc_score(label, t_predict)
        acc_test.append(t_acc)
        loss_test.append(t_loss.numpy())
    print("Test: Loss {:.4f} | Acc {:.4f}".format(np.mean(loss_test), np.mean(acc_test)))
    accs_all.append(np.mean(acc_test))
print("ACC: {:.4f}±{:.4f}".format(np.mean(accs_all), np.std(accs_all)))
