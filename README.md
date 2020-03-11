![avatar](att.png)

[![Build Status](https://travis-ci.org/kyzhouhzau/fennlp.svg?branch=master)](https://travis-ci.org/kyzhouhzau/fennlp/branches)
[![PyPI version](https://badge.fury.io/py/fennlp.svg)](https://badge.fury.io/py/fenlp)
[![GitHub version](https://badge.fury.io/gh/kyzhouhzau%2Ffennlp.svg)](https://badge.fury.io/gh/kyzhouhzau%2Ffennlp)
[![Maintainability](https://api.codeclimate.com/v1/badges/d587092245542684c80b/maintainability)](https://codeclimate.com/github/kyzhouhzau/fennlp/maintainability)
[![License](https://img.shields.io/github/license/kyzhouhzau/fennlp)](https://github.com/kyzhouhzau/fennlp/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/kyzhouhzau/fennlp/badge.svg)](https://coveralls.io/github/kyzhouhzau/fennlp)

# Package description
An out-of-the-box NLP toolkit can easily help you solve tasks such as
Entity Recognition, Text Classification, Relation Extraction and so on.
Currently it contain the following models (see "tests" dictionary for more details):
* BERT (tf2.0 layer, Chinese and English Version)
* BERT-NER (Chinese Version, 中文糖尿病标注数据集)
* BERT-CRF-NER (Chinese Version, 中文糖尿病标注数据集)
* BERT-Sentence-Classification(Chinese Version, 新闻标题短文本分类)
* TextCNN(Chinese and English Version, 新闻标题短文本分类)
* GCN (2 Layer, CORA data set)
* TuckER (English Version, WN18 data set)

Use BERT as tensorflow2.0's layer, See tests dictionary for more details。


# Status
2020/3/8: add test example "run_tucker.py" for train TuckER on WN18.

2020/3/3: add test example "tran_text_cnn.py" for train TextCNN model. 

2020/3/2: add test example "train_bert_classification.py" for text classification based on bert.

2020/2/26: add GCN example on cora data.

2020/2/25: add test example "bert_ner_train.py" and "bert_ner_test.py".


# Requirement
* tensorflow>=2.0
* typeguard
* gensim

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

# For NER：

## Input
* put train, valid and test file in "Input" dictionary.
* data format: reference data in  "tests\NER\InputNER\train"

    e.g. "拮 抗 RANKL 对 破 骨 细 胞 的 作 用 。	O O O O B-Anatomy I-Anatomy I-Anatomy E-Anatomy O O O O"
    
    For each line in train contains two parts, the first part "拮 抗 RANKL 对 破 骨 细 胞 的 作 用 。" is a sentence.
    The second part "O O O O B-Anatomy I-Anatomy I-Anatomy E-Anatomy O O O O" is the tag for each word in the sentence.
    Both of them use '\t' to concatenate.

### 1、bert

```python
from fennlp.models import bert
bert = bert.BERT()
```

```
python bert_ner_train.py
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bert (BERT)                  multiple                  101677056 
_________________________________________________________________
dense (Dense)                multiple                  35374     
=================================================================
Total params: 101,712,430
Trainable params: 101,712,430
Non-trainable params: 0
_________________________________________________________________
```

### 2、bert + crf
```python
from fennlp.models import bert
from fennlp.metrics.crf import CrfLogLikelihood
bert = bert.BERT()
crf = CrfLogLikelihood()
```

```
python bert_ner_crf_train.py
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bert (BERT)                  multiple                  101677056 
_________________________________________________________________
dense (Dense)                multiple                  35374     
_________________________________________________________________
crf (CrfLogLikelihood)       multiple                  2116      
=================================================================
Total params: 101,714,546
Trainable params: 101,714,546
Non-trainable params: 0
_________________________________________________________________
```

Using the default parameters, we get the following results on "中文糖尿病标注数据集" valid data.

|model|  macro-F1   | macro-P  | macro-R  | ACC  |
| ------- |  -------| -------  | -------  | ------- |
| bert+crf | 0.6288  | 0.6507 | 0.6493 | 0.9835 |
|   bert   |         |        |        |        |


# For Sentence Classfication

## Input
* put train, valid and test file in "Input" dictionary.
* data format: reference data in "\tests\CLS\BERT( or TextCNN)\Input".

    e.g. "作 为 地 球 上 曾 经 最 强 的 拳 王 之 一 ， 小 克 里 琴 科 谈 自 己 是 否 会 复 出    2"
    
    For each line in train(test,valid) contains two parts, the first part "作 为 地 球 上 曾 经 最 强 的 拳 王 之 一 ，
    小 克 里 琴 科 谈 自 己 是 否 会 复 出" is the sentence, and second part "2" is the label.

### 1、bert

```python
from fennlp.models import bert
bert = bert.BERT()
```

``` 
python train_bert_classification.py
```

### 2、TextCNN

```python
from fennlp.models import TextCNN
model = TextCNN.TextCNN()
```

``` 
python train_text_cnn.py
```

```
TODO: use "WordPiece embedding" to Initialize word embedding.
```
For more detail about reference [WordPiece](https://mp.weixin.qq.com/s/Il8sh66TUCEPskbypDZLAg) 

# For GCN：

## Input
data format: see files in "tests/GCN/data/README.md" for more detail.


```python
from fennlp.models import GCN
model = GCN.GCN2Layer()
```

```
python train_gcn.py
```







