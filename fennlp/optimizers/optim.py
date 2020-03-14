#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
参考：
https://github.com/CyberZHG/keras-bert
https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/backend.py#L1561-L1563
https://github.com/bojone/bert4keras
"""
import tensorflow as tf


class Adam(tf.keras.optimizers.Optimizer):
    def __init__(self,
                 learning_rate=0.01,
                 min_lr=1e-9,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 bias_correction=True,
                 name='Adam',
                 **kwargs):
        """
        :param warmup_steps: 学习率在指定的步数线性增长到目标学习率
        :param learning_rate:
        :param beta_1:
        :param beta_2:
        :param epsilon:
        :param bias_correction:
        :param name:
        :param kwargs:
        """
        super(Adam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('min_lr', min_lr)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or tf.keras.backend.epislon()
        self.bias_correction = bias_correction

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply_op(self, grad, var, indices=None):
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        min_lr = self._get_hyper('min_lr', var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t_power = tf.math.pow(beta_1_t, local_step)
        beta_2_t_power = tf.math.pow(beta_2_t, local_step)

        if indices is None:
            # update 与 state_ops.assign(x,new_x)相同均是赋值含义
            m_t = tf.keras.backend.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = tf.keras.backend.update(v, beta_2_t * v + (1 - beta_2_t) * grad ** 2)
        else:
            mv_ops = [tf.keras.backend.update(m, beta_1_t * m), tf.keras.backend.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(m, indices,
                                                 (1 - beta_1_t) * grad)
                v_t = self._resource_scatter_add(v, indices,
                                                 (1 - beta_2_t) * grad ** 2)

        with tf.control_dependencies([m_t, v_t]):
            # 偏差修正
            if self.bias_correction:
                m_t = m_t / (1. - beta_1_t_power)
                v_t = v_t / (1. - beta_2_t_power)
            var_t = var - lr_t * m_t / (tf.keras.backend.sqrt(v_t) + epsilon_t)
            return tf.keras.backend.update(var, var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply_op(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply_op(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
        }
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

