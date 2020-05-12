#! encoding:utf-8
import tensorflow as tf
import numpy as np
import time
from fennlp.datas import TUDataset
from fennlp.metrics import Losess, Metric
from fennlp.models import GINLayer
from fennlp.gnn.utils import merge_batch_graph

dim = 64
num_class = 2
drop_rate = 0.5
epoch = 300
lr = 0.001
split = 10  # 10-fold

tf.random.set_seed(1124)
accs_all = []
dataloader = TUDataset(name="PROTEINS", split=split)

for block_index in range(split):

    model = GINLayer(dim, num_class, drop_rate)

    optimize = tf.optimizers.Adam(lr)

    cross_entropy = Losess.MaskSparseCategoricalCrossentropy()
    acc_score = Metric.SparseAccuracy()

    train_data, test_data = dataloader.load(batch_size=128, block_index=block_index)
    for i in range(epoch):
        t = time.time()
        loss_all = []
        acc_all = []
        for x, y, edge_index, edge_attr, batch in train_data:
            x, y, edge_index, edge_attr, batch = merge_batch_graph(x, y, edge_index,
                                                                   edge_attr, batch)

            with tf.GradientTape() as tape:
                predict = model(x, edge_index, batch, training=True)
                loss = cross_entropy(y, predict)
                acc = acc_score(y, predict)
                loss_all.append(loss.numpy())
                acc_all.append(acc)

            grads = tape.gradient(loss, model.trainable_variables)
            optimize.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        if i != 0 and i % 50 == 0:
            lr_rate = optimize._get_hyper("learning_rate")
            optimize._set_hyper('learning_rate', lr_rate * 0.5)
        print("Fold: {:.0f} | Train: Epoch {} | Loss {:.4f} | Acc {:.4f} | Time {:.4f}".format(block_index, i, np.mean(loss_all), np.mean(acc_all),time.time()-t))

    # ------------------------------------------------------------------------
    loss_test = []
    acc_test = []
    for x, y, edge_index, edge_attr, batch in test_data:
        x, y, edge_index, edge_attr, batch = merge_batch_graph(x, y, edge_index,
                                                               edge_attr, batch)
        t_predict = model.predict(x, edge_index, batch, training=False)
        t_loss = cross_entropy(y, t_predict)
        t_acc = acc_score(y, t_predict)
        acc_test.append(t_acc)
        loss_test.append(t_loss.numpy())
    print("Fold: {:.0f} | Test: Loss {:.4f} | Acc {:.4f}".format(block_index,np.mean(loss_test), np.mean(acc_test)))
    accs_all.append(np.mean(acc_test))

print("ACC: {:.4f}±{:.4f}".format(np.mean(accs_all), np.std(accs_all)))
