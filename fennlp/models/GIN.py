#! encoding="utf-8"
import tensorflow as tf

from fennlp.gnn.GINConv import GINConvolution
from fennlp.gnn.utils import GNNInput

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
ReLU = tf.keras.layers.ReLU


class GINLayer(tf.keras.Model):
    def __init__(self, dim, num_classes, drop_rate, **kwargs):
        super(GINLayer, self).__init__(**kwargs)
        nn1 = Sequential([Dense(dim), ReLU(), Dense(dim)])
        self.conv1 = GINConvolution(nn1,eps=0)
        nn2 = Sequential([Dense(num_classes), ReLU(), Dense(num_classes)])
        self.conv2 = GINConvolution(nn2,eps=0)
        self.drop_out1 = tf.keras.layers.Dropout(drop_rate)
        self.drop_out2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, node_embeddings, adjacency_lists, training=True):
        node_embeddings = self.drop_out1(node_embeddings, training=training)
        x = tf.nn.relu(self.conv1(GNNInput(node_embeddings, adjacency_lists)))
        x = self.drop_out2(x, training=training)
        x = self.conv2(GNNInput(x, adjacency_lists))
        return tf.math.softmax(x, -1)

    def predict(self, node_embeddings, adjacency_lists, training=False):
        return self(node_embeddings, adjacency_lists, training)
