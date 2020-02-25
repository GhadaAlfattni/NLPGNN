#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import re
import time
import tensorflow as tf
import collections
import numpy as np
from tensorflow.python.keras import backend as K

version = tf.__version__
Version_float = float('.'.join(version.split('.')[:2]))


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    # return tf.keras.initializers.get("truncated_normal")
    return tf.keras.initializers.TruncatedNormal(initializer_range)


def reshape_to_matrix(tensor):
    if len(tensor.shape) == 0:
        return tensor
    dim = tensor.shape[-1]
    tensor_2d = tf.reshape(tensor, [-1, dim])
    return tensor_2d


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor, orig_dims + [width])


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """
    Create 3D attention mask from a 2D tensor mask.
    from_tensor [B,F,D]
    to_mask [B,T]
    """
    assert_rank(from_tensor, [2, 3])
    from_shape = get_shape_list(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_shape = get_shape_list(to_mask)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32
    )

    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32
    )

    mask = broadcast_ones * to_mask
    return mask


def assert_rank(tensor, expected_rank, name=None):
    # if name is None:
    #     name = tensor.name
    excepted_rank_dict = {}
    if isinstance(expected_rank, int):
        excepted_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            excepted_rank_dict[x] = True
    actual_rank = tensor.shape.ndims
    if actual_rank not in excepted_rank_dict:
        raise (
            "For tensor {} , the actual rank {} is not equal"
            "to expected rank {}".format(name, actual_rank, str(expected_rank))
        )


def gelu(x):
    import numpy as np
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(activation_string):
    "map string to activation function"
    if not isinstance(activation_string, str):
        return activation_string
    if not activation_string:
        return None
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "gelu":
        return gelu
    elif act == "relu":
        return tf.nn.relu
    elif act == "tanh":
        return tf.nn.tanh
    else:
        raise ValueError("Unsupport activation:%s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    initialized_variable_names = {}
    name_to_variable = collections.OrderedDict()
    for var in tvars:
        print(var.name)
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(init_checkpoint)
    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)


def init_weights_from_checkpoint(model, checkpoint_file, num_layer,pooler):
    def _loader(checkpoint_file):
        def _loader(name):
            return tf.train.load_variable(checkpoint_file, name)

        return _loader

    loader = _loader(checkpoint_file)
    weights = [
        loader("bert/embeddings/word_embeddings"),
        loader("bert/embeddings/token_type_embeddings"),
        loader("bert/embeddings/position_embeddings"),
        loader("bert/embeddings/LayerNorm/gamma"),
        loader("bert/embeddings/LayerNorm/beta"),
    ]
    encoder = []
    for i in range(num_layer):
        encoder.extend([
            loader("bert/encoder/layer_{}/attention/self/query/kernel".format(i)),
            loader("bert/encoder/layer_{}/attention/self/query/bias".format(i)),
            loader("bert/encoder/layer_{}/attention/self/key/kernel".format(i)),
            loader("bert/encoder/layer_{}/attention/self/key/bias".format(i)),
            loader("bert/encoder/layer_{}/attention/self/value/kernel".format(i)),
            loader("bert/encoder/layer_{}/attention/self/value/bias".format(i)),
            loader("bert/encoder/layer_{}/attention/output/dense/kernel".format(i)),
            loader("bert/encoder/layer_{}/attention/output/dense/bias".format(i)),
            loader("bert/encoder/layer_{}/intermediate/dense/kernel".format(i)),
            loader("bert/encoder/layer_{}/intermediate/dense/bias".format(i)),
            loader("bert/encoder/layer_{}/output/dense/kernel".format(i)),
            loader("bert/encoder/layer_{}/output/dense/bias".format(i)),
            loader("bert/encoder/layer_{}/output/LayerNorm/gamma".format(i)),
            loader("bert/encoder/layer_{}/output/LayerNorm/beta".format(i)),
            loader("bert/encoder/layer_{}/attention/output/LayerNorm/gamma".format(i)),
            loader("bert/encoder/layer_{}/attention/output/LayerNorm/beta".format(i)),
        ])

    weights.extend(encoder)
    if pooler:
        weights.extend([loader("bert/pooler/dense/kernel"),
                    loader("bert/pooler/dense/bias")])

    variables = model.get_layer("bert").variables
    varname = [var.name for var in variables]
    # weight = {name for name in weights}
    for i, var in enumerate(varname):
        print("INIT WEIGHTS {}".format(var), i)

    model.get_layer("bert").set_weights(weights)
    del weights


# calculate function time
def timeit(func):
    def wrapper(*args, **kwargs):
        """
        :param args: 
        :param kwargs: 
        :return:
        """
        start = time.clock()
        result = func(*args, **kwargs)
        end = time.clock()
        print("该函数用时{}".format(end - start))
        return result

    return wrapper


def get_shape_list(tensor, expected_rank=None, name=None):
    # if name is None:
    #     name = tensor.name
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:

        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)
