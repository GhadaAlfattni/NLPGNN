import tensorflow as tf
from fennlp.models import bert
from fennlp.optimizers import optim
from fennlp.tools import init_weights_from_checkpoint
from fennlp.datas.checkpoint import LoadCheckpoint
from fennlp.datas.dataloader import ZHTFWriter, ZHTFLoader
from fennlp.metrics import Metric, Losess

# 载入参数
load_check = LoadCheckpoint()
param, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
param["batch_size"] = 2
param["maxlen"] = 10
param["label_size"] = 15

