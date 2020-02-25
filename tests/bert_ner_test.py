import tensorflow as tf
from fennlp.models import bert
from fennlp.datas.checkpoint import LoadCheckpoint
from fennlp.datas.dataloader import ZHTFWriter, NERLoader
from fennlp.metrics import Metric
from fennlp.metrics.crf import crf_decode,crf_log_likelihood

# 载入参数
load_check = LoadCheckpoint()
param, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
param["batch_size"] = 2
param["maxlen"] = 100
param["label_size"] = 47


# 构建模型
class BERT_NER(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(BERT_NER, self).__init__(**kwargs)
        self.batch_size = param["batch_size"]
        self.maxlen = param["maxlen"]
        self.label_size = param["label_size"]

        self.bert = bert.BERT(param)

        self.dense = tf.keras.layers.Dense(self.label_size, activation="relu")

    def call(self, inputs, is_training=True):
        bert = self.bert(inputs, is_training)
        sequence_output = bert.get_sequence_output()  # batch,sequence,768
        output = self.dense(sequence_output)
        output = tf.reshape(output, [self.batch_size, self.maxlen, -1])
        output = tf.math.softmax(output, axis=-1)
        return output

    def predict(self, inputs, is_training=False):
        predict = self(inputs, is_training=is_training)
        return predict


model = BERT_NER(param)

model.build(input_shape=(3, param["batch_size"], param["maxlen"]))

model.summary()

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = ZHTFWriter(param["maxlen"], vocab_file,
                    modes=["Valid"], check_exist=True)

ner_load = NERLoader(param["maxlen"], param["batch_size"], epoch=3)

# Metrics
f1score = Metric.SparseF1Score("macro")
precsionscore = Metric.SparsePrecisionScore("macro")
recallscore = Metric.SparseRecallScore("macro")
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('./save'))
# For test model
# print(dir(checkpoint))
Batch = 0
for X, token_type_id, input_mask, Y in ner_load.load_valid():
    output = model.predict([X, token_type_id, input_mask])#[batch_size, max_length,label_size]
    predict = tf.argmax(output,-1)


    print("Sentence", writer.convert_id_to_vocab(tf.reshape(X,[-1]).numpy()))

    print("Label", writer.convert_id_to_label(tf.reshape(predict,[-1]).numpy()))
