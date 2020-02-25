#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from FenNLP.tools import init_weights_from_checkpoint,generator
import warnings
warnings.filterwarnings("ignore")
class Trainer(object):
    @staticmethod
    # @tf.function
    def train(loader, model, losser, optimizer,config):
        if config.GPUNUMS<=1:
            model.build(tf.TensorShape([3, config.batch_size, config.maxlen]))
        else:
            with tf.device("/cpu:0"):
                model.build(tf.TensorShape([3, config.batch_size, config.maxlen]))
            model = tf.keras.utils.multi_gpu_model(model,config.GPUNUMS)

        if config.freeze:
            model.layers[-1].trainable = False

        model.summary()
        if config.init_checkpoint != None:
            init_weights_from_checkpoint(model,config.init_checkpoint,config.num_hidden_layers)
            print("\n******INIT WEIGHTS FROM CHECKPOINT SUCCESS******\n")

        if config.is_training:
            train_generator = generator(loader, config, "train")
            for input,target in train_generator.generate_arrays():
                with tf.GradientTape() as tape:
                    logits = model(input,target)
                    print(target)
                    print(tf.argmax(logits,-1).numpy())
                    loss= losser(target,logits)
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,-1),target),tf.float32))
                    print("LOSS:{:.3f}\tACC:{:.3f}".format(loss.numpy(),accuracy.numpy()))
                gradients = tape.gradient(loss, model.trainable_variables)
                (gradients, _) = tf.clip_by_global_norm(gradients, clip_norm=1.0)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # model.save_weights(config.output_model)

    @staticmethod
    def dev(loader,model,losser,config,batchsize):
        model.load_weights(config.output_model)
        model.is_training=False
        model.batch_size = batchsize
        for tinput, tmask, ttarget,tsegment in loader.load(config =config,dev =True):
            tlogits = model(tinput, tmask, tsegment)
            loss = losser(tf.keras.backend.one_hot(ttarget,config.num_class), tlogits)
            print("LOSS:{}".format(loss))

    @staticmethod
    def predict(loader,model,config,output_file,batchsize):
        model.load_weights(config.output_model)
        model.is_training = False
        model.batch_size = batchsize
        wf = open(output_file,'w')
        for tinput, tmask, ttarget, tsegment in loader.load(config=config, dev=False):
            tlogits = model(tinput, tmask, tsegment)
            predicts = tf.math.argmax(tlogits,axis=1)
            for label in predicts:
                wf.write(label+'\n')
