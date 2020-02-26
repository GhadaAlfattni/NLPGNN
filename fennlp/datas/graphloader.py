#! encoding:utf-8

import tensorflow as tf
import numpy as np
from scipy import sparse


class GCNLoader():
    def __init__(self, base_path="data", dataset="cora"):
        self.base_path = base_path
        self.dataset = dataset
        print("Loading {} dataset...".format(dataset))

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = (1 / rowsum).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sparse.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def convert_2_sparse_tensor(self, sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
        values = sparse_matrix.data
        shape = sparse_matrix.shape
        indices = np.array([[row, col] for row, col in zip(sparse_matrix.row, sparse_matrix.col)], dtype=np.int64)
        return tf.sparse.SparseTensor(indices, values, shape)

    def load(self):
        idx_features_labels = np.genfromtxt("{}/{}.content".format(self.base_path, self.dataset),
                                            dtype=np.dtype(str))
        features = sparse.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        print

        # 构建图
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/{}.cites".format(self.base_path, self.dataset),
                                        dtype=np.int32)
        # [[1,2],
        #  [22,23]]
        # N*2
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)

        # 构建对称邻接矩阵
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = self.normalize(features)
        adj = self.normalize(adj + sparse.eye(adj.shape[0]))

        features = tf.constant(np.array(features.todense()))

        labels = tf.constant(np.where(labels)[1])
        adj = self.convert_2_sparse_tensor(adj)


        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        return features, adj, labels, idx_train, idx_val, idx_test


# loader = GCNLoader(r"C:\Users\Administrator\Desktop\fennlp\tests\GCN\data")
# features, adj, labels, idx_train, idx_val, idx_test = loader.load()