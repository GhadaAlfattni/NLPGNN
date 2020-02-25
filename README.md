# fennlp

An out-of-the-box NLP toolkit can easily help you solve tasks such as entity recognition, relationship extraction, text classfication and so on.

# Require
tensorflow>=2.0

# Usage
1. clone source
```
git clone https://github.com/kyzhouhzau/fennlp.git
```
2. install package
```
python setup.py install
```
3.run test file
```
cd tests
```
```
python bert_ner_train.py
```

# For NER：

## input
* build dictionary "InputNER" and put train, Valid, test file in it.
* Data format for train Valid and test :
Reference data in  “tests\InputNER\train”

Use BERT as an tensorflow layer

```python
from fennlp.models import bert
self.bert = bert.BERT(param)
```
## run ner
```
python bert_ner_train.py
```




# Status
2020/2/25: add test example "bert_ner_train.py" and "bert_ner_test.py"



