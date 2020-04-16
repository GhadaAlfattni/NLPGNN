#! encoding:utf-8
import time

import tensorflow as tf

from fennlp.datas import graphloader
from fennlp.metrics import Losess, Metric
from fennlp.models import GCN

hidden_dim = 16
num_class = 7
drop_rate = 0.5
epoch = 200
early_stopping = 10

loader = graphloader.GCNLoader("cora")
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = loader.load()

model = GCN.GCNLayer(hidden_dim, num_class, drop_rate)

optimizer = tf.keras.optimizers.Adam(0.01)

crossentropy = Losess.MaskCategoricalCrossentropy(use_mask=True)
accscore = Metric.MaskAccuracy()
# ---------------------------------------------------------
# For train
wait = 0
best_see_loss = 9999
t = time.time()
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        predict = model(features, adj, training=True)
        loss = crossentropy(y_train, predict, train_mask)
        #
        predict_v = model.predict(features, adj)
        see_loss = crossentropy(y_val, predict_v, val_mask)
        if see_loss < best_see_loss:
            best_see_loss = see_loss
        else:
            if wait >= early_stopping:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1
        acc = accscore(y_val, predict, val_mask)
        print("Epoch {} | Loss {:.4f} | Acc {:.4f} | Time {:.0f}".format(epoch, see_loss.numpy(), acc, time.time() - t))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
# ------------------------------------------------------
# For test
output = model.predict(features, adj)
acc = accscore(y_test, predict, test_mask)
loss = crossentropy(y_test, predict, test_mask)
print("Test Loss {:.4f} | ACC {:.4f} | Time {:.0f}".format(loss.numpy(), acc, time.time() - t))
