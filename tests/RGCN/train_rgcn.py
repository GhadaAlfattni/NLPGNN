#! encoding="utf-8"
from fennlp.datas.graphloader import RGCNLoader
from fennlp.models import RGCN
negative_sample = 1
batch_size = 10
split_size = 0.5
num_bases = 4
reg_ratio = 1e-2
dropout = 0.2
data = RGCNLoader("data", "FB15k-237")
entity2id, relation2id, train_triplets, valid_triplets, test_triplets = data.load_data()
data.generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, len(entity2id), len(relation2id),
                                          negative_sample)

print(data.entity, data.edge_index, data.edge_type)
print(len(entity2id))
print(len(relation2id))
model = RGCN.RGCNDistmult(len(entity2id), len(relation2id), num_bases, dropout)

entity_embedding = model(data.entity, data.edge_index, data.edge_type, data.edge_norm)

loss = model.score_loss(entity_embedding, data.samples, data.labels) + reg_ratio * model.reg_loss( entity_embedding)
