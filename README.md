![avatar](att.png)

[![Build Status](https://travis-ci.org/kyzhouhzau/fennlp.svg?branch=master)](https://travis-ci.org/kyzhouhzau/fennlp/branches)
[![PyPI version](https://badge.fury.io/py/fennlp.svg)](https://badge.fury.io/py/fenlp)
[![GitHub version](https://badge.fury.io/gh/kyzhouhzau%2Ffennlp.svg)](https://badge.fury.io/gh/kyzhouhzau%2Ffennlp)
[![Maintainability](https://api.codeclimate.com/v1/badges/d587092245542684c80b/maintainability)](https://codeclimate.com/github/kyzhouhzau/fennlp/maintainability)
[![License](https://img.shields.io/github/license/kyzhouhzau/fennlp)](https://github.com/kyzhouhzau/fennlp/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/kyzhouhzau/fennlp/badge.svg)](https://coveralls.io/github/kyzhouhzau/fennlp)

# Package description
An out-of-the-box NLP toolkit can easily help you solve tasks such as
entity recognition, relationship extraction, text classfication and so on.
Currently it contain the following models (see "tests" dictionary for more details):
* BERT (tf2.0 layer, Chinese and English Version)
* BERT-NER (Chinese Version, 中文糖尿病标注数据集)
* BERT-CRF-NER (Chinese Version, 中文糖尿病标注数据集)
* BERT-Sentence-Classification(Chinese Version, 新闻标题短文本分类)
* TextCNN(Chinese and English Version, 新闻标题短文本分类)
* GCN (2 layer)
* TuckER (English Version, WN18)

# Status
2020/3/8: add test example "run_tucker.py" for train TuckER on WN18.

2020/3/3: add test example "tran_text_cnn.py" for train TextCNN model. 

2020/3/2: add test example "train_bert_classification.py" for text classification based on bert.

2020/2/26: add GCN example on cora data.

2020/2/25: add test example "bert_ner_train.py" and "bert_ner_test.py".


# Requirement
* tensorflow>=2.0

# Usage

1. clone source
```
git clone https://github.com/kyzhouhzau/fennlp.git
```
2. install package
```
python setup.py install
```
3. run model
```
python bert_ner_train.py
```
# For Sentence Classfication

## Input
* put train, valid and test file in "Input" dictionary.
* data format: reference data in "\tests\CLS\BERT( or TextCNN)\Input".

e.g. "作 为 地 球 上 曾 经 最 强 的 拳 王 之 一 ， 小 克 里 琴 科 谈 自 己 是 否 会 复 出    2"

For each line in train(test,valid) contains two parts, the first part "作 为 地 球 上 曾 经 最 强 的 拳 王 之 一 ，
小 克 里 琴 科 谈 自 己 是 否 会 复 出" is the sentence, and second part "2" is the label.

### BERT based

```python
from fennlp.models import bert
self.bert = bert.BERT(param)
bert = bert(inputs, is_training)
sequence_output = bert.get_pooled_output()
```

``` 
python train_bert_classification.py
```

### TextCNN

```python
from fennlp.models import TextCNN
model = TextCNN.TextCNN(maxlen, vocab_size, embedding_dims, class_num)
```

``` 
python train_text_cnn.py
```
```
TODO: use "WordPiece embedding" to Initialize word embedding.
```
For more detail about reference [WordPiece](https://mp.weixin.qq.com/s/Il8sh66TUCEPskbypDZLAg) 

# For NER：

## Input
* put train, valid and test file in "Input" dictionary.
* data format: reference data in  "tests\NER\InputNER\train"

e.g. "拮 抗 RANKL 对 破 骨 细 胞 的 作 用 。	O O O O B-Anatomy I-Anatomy I-Anatomy E-Anatomy O O O O"

For each line in train contains two parts, the first part "拮 抗 RANKL 对 破 骨 细 胞 的 作 用 。" is a sentence.
The second part "O O O O B-Anatomy I-Anatomy I-Anatomy E-Anatomy O O O O" is the tag for each word in the sentence.
Both of them use '\t' to concatenate.

Use BERT as an tensorflow2.0's layer, See tests for more detail。

### without crf

```python
from fennlp.models import bert
bert = bert.BERT(param)
bert = bert(inputs, is_training)
sequence_output = bert.get_sequence_output()
```

```
python bert_ner_train.py
```

### with crf
```python
bert = self.bert([input_ids, token_type_ids, input_mask], is_training)
sequence_output = bert.get_sequence_output()  # batch,sequence,768
predict = self.dense(sequence_output)
predict = tf.reshape(predict, [self.batch_size, self.maxlen, -1])
# crf
log_likelihood, transition = self.crf(predict, Y, sequence_lengths=tf.reduce_sum(input_mask, 1))
loss = tf.math.reduce_mean(-log_likelihood)
predict, viterbi_score = self.crf.crf_decode(predict, transition, sequence_length=tf.reduce_sum(input_mask, 1))
```

```
python bert_ner_crf_train.py
```

# For GCN：

## Input
data format: see files in "tests/GCN/data/README.md" for more detail.


```python
from fennlp.models import GCN
model = GCN.GCN2Layer(_HIDDEN_DIM, _NUM_CLASS, _DROP_OUT_RATE)
```

```
python train_gcn.py
```







