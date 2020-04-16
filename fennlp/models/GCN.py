#! encoding="utf-8"
import tensorflow as tf

from fennlp.gnn.GCNConv import GraphConvolution
from fennlp.gnn.utils import GNNInput, add_self_loop


class GCNLayer(tf.keras.Model):
    def __init__(self, hidden_dim, num_class, penalty=5e-4, dropout_rate=0.5, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.gc1 = GraphConvolution(hidden_dim,
                                    regularizer=tf.keras.regularizers.l2(penalty),
                                    name='gcn1')
        self.gc2 = GraphConvolution(num_class, name='gcn2')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, node_embeddings, adjacency_lists, training=True):
        adjacency_loop = [add_self_loop(adjacency_lists, len(node_embeddings))]
        x = self.gc1(GNNInput(node_embeddings, adjacency_loop), training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.gc2(GNNInput(x, adjacency_loop), training)
        return tf.math.softmax(x, 1)

    def predict(self, node_embeddings, adjacency_lists, training=False):
        return self(node_embeddings, adjacency_lists, training)
