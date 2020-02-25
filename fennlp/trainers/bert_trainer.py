#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import os
import warnings
import tensorflow as tf
from tools import init_weights_from_checkpoint,generator
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def scheduler(epochs,lr):
    return float(lr * tf.math.exp(0.1 * (10 - epochs)))

class Trainer(object):
    @staticmethod
    def train(loader, model, losser, optimizer, config):
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
            print("\n******INIT WEIGHTS FROM CHECKPOINT******\n")
            init_weights_from_checkpoint(model, config.init_checkpoint, config.num_hidden_layers)
            print("\n******INIT WEIGHTS FROM CHECKPOINT SUCCESS******\n")

        model.compile(
            optimizer=optimizer,
            loss=losser,
            metrics=['accuracy']
        )

        if config.is_training:
            # callbacks
            checkpoint_path = config.output_model+"/cp-{epoch:04d}.ckpt"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             verbose=1)

            # es_callback = tf.keras.callbacks.EarlyStopping()
            # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config.logs,
            #                                              update_freq="batch",
            #                                              write_images=True)

            train_generator = generator(loader, config, "train")
            dev_generator = generator(loader, config, "dev")
            model.fit_generator(
                generator=train_generator.generate_arrays(),
                steps_per_epoch=train_generator.bindex,
                epochs=config.epoch,
                # validation_steps=dev_generator.bindex,
                callbacks=[cp_callback],
                workers=1,
                # use_multiprocessing=True,
            )

    @staticmethod
    def evaluate(loader,model,config):
        config.is_training = False
        model.load_weights(config.output_model)
        dev_generate = generator(loader,config,"dev")
        model.evaluate_generator(
            generator=dev_generate.generate_arrays(),
            steps=dev_generate.bindex
        )

    @staticmethod
    def predict(loader, model, config):
        model.build(tf.TensorShape([3, config.batch_size, config.maxlen]))
        model.summary()
        config.is_training=False
        model.load_weights(config.output_model)
        print(model.trainable_variables)
        test_generate = generator(loader,config,"test")
        model.predict_generator(
            generator=test_generate.generate_arrays(),
            steps=test_generate.bindex
        )