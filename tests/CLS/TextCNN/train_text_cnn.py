#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf
from fennlp.models import TextCNN
from fennlp.optimizers import optim
from fennlp.metrics import Metric, Losess
from fennlp.datas.dataloader import ZHTFWriter, ZHTFLoader

maxlen = 50
batch_size = 128
embedding_dims = 100
vocab_file = "Input/vocab.txt"
vocab_size = 21128  # line in vocab.txt
class_num = 15

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = ZHTFWriter(maxlen,
                    vocab_file,
                    modes=["train"],
                    task='cls',
                    check_exist=True)

load = ZHTFLoader(maxlen,
                  batch_size,
                  task='cls',
                  epoch=3)

model = TextCNN.TextCNN(maxlen, vocab_size, embedding_dims, class_num)

# 构建优化器
optimizer = optim.Adam(learning_rate=0.1, min_lr=0.0001, decay_steps=500)

# 构建损失函数
mask_sparse_categotical_loss = Losess.MaskSparseCategoricalCrossentropy(from_logits=False)

f1score = Metric.SparseF1Score(average="macro")
precsionscore = Metric.SparsePrecisionScore(average="macro")
recallscore = Metric.SparseRecallScore(average="macro")
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory="./save",
                                     checkpoint_name="model.ckpt",
                                     max_to_keep=3)
Batch = 0
for X, token_type_id, input_mask, Y in load.load_train():
    with tf.GradientTape() as tape:
        predict = model(X)
        loss = mask_sparse_categotical_loss(Y, predict, use_mask=True)
        f1 = f1score(Y, predict)
        precision = precsionscore(Y, predict)
        recall = recallscore(Y, predict)
        accuracy = accuarcyscore(Y, predict)
        if Batch % 100 == 0:
            print("Batch:{}\tloss:{:.4f}".format(Batch, loss.numpy()))
            print("Batch:{}\tacc:{:.4f}".format(Batch, accuracy))
            print("Batch:{}\tprecision{:.4f}".format(Batch, precision))
            print("Batch:{}\trecall:{:.4f}".format(Batch, recall))
            print("Batch:{}\tf1score:{:.4f}".format(Batch, f1))
            manager.save(checkpoint_number=Batch)
    grads_bert = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads_bert, model.variables))
    Batch += 1
