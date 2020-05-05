#! encoding="utf-8"

from fennlp.gnn.GINConv import GINConvolution
from fennlp.gnn.utils import *
import tensorflow as tf

Dense = tf.keras.layers.Dense
Drop = tf.keras.layers.Dropout
BatchNorm = tf.keras.layers.BatchNormalization


class MLP(tf.keras.Model):
    def __init__(self, dim, layers, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.Denses = []
        for i in range(layers):
            self.Denses.append(Dense(dim, activation="relu"))
        self.batch_norm = BatchNorm()

    def call(self, x, training=True):
        for i, layer in enumerate(self.Denses):
            x = layer(x)
        x = self.batch_norm(x, training)
        return x


class GINLayer(tf.keras.Model):
    def __init__(self, dim=32, num_classes=2, drop_rate=0.5, **kwargs):
        super(GINLayer, self).__init__(**kwargs)
        nn1 = MLP(dim, 2)
        self.conv1 = GINConvolution(nn1, eps=0)
        self.bn1 = tf.keras.layers.BatchNormalization()

        nn2 = MLP(dim, 2)
        self.conv2 = GINConvolution(nn2, eps=0)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.fc1 = Dense(dim)
        self.fc2 = Dense(num_classes)

        self.drop_out2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, node_embeddings, edge_index, batch, training=True):
        edge_index = [edge_index]
        x = tf.nn.relu(self.conv1(GNNInput(node_embeddings, edge_index), training))
        x = tf.nn.relu(self.conv2(GNNInput(x, edge_index), training))
        x = batch_read_out(x, batch)

        x = tf.nn.relu(self.fc1(x))
        x = self.drop_out2(x, training=training)
        x = self.fc2(x)
        return tf.math.softmax(x, -1)

    def predict(self, node_embeddings, edge_index, batch, training=False):
        return self(node_embeddings, edge_index, batch, training)
