#!encoding:utf-8
import requests
import sys
import zipfile
import os
import json


class LoadCheckpoint(object):
    def __init__(self, langurage='zh'):
        self.lg = langurage
        if langurage == "zh":

            url = "https://storage.googleapis.com/bert_models/" \
                  "2018_11_03/chinese_L-12_H-768_A-12.zip"
        elif langurage == "en":
            url = "https://storage.googleapis.com/bert_models/" \
                  "2018_10_18/uncased_L-12_H-768_A-12.zip"
        self.url = url
        self.size = self.getsize()
        self.unzip()

    def getsize(self):
        try:
            r = requests.head(self.url)
            size = r.headers.get("Content-Length")
            return int(size)
        except:
            print("Failed Download!")
            exit()

    def bar(self, num, total):
        rate = num / total
        rate_num = int(rate * 100)
        if rate_num == 100:
            r = '\r%s>%d%%\n' % ('=' * int(rate_num / 3), rate_num,)  # 控制等号输出数量，除以3,表示显示1/3
        else:
            r = '\r%s>%d%%' % ('=' * int(rate_num / 3), rate_num,)
        sys.stdout.write(r)
        sys.stdout.flush()

    def download(self, chunk_size=1024):
        num = 0
        response = requests.get(self.url, stream=True)
        with open(self.url.split('/')[-1], 'wb') as wf:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    wf.write(chunk)
                    num += chunk_size
                    self.bar(num, self.size)

    def unzip(self):
        filename = self.url.split('/')[-1]
        if not os.path.exists(filename):
            open(filename, 'w').close()
        if os.path.getsize(filename) != self.size:
            print("Download and unzip: {}".format(filename))
            self.download()
        zip_file = zipfile.ZipFile(filename)
        zip_file.extractall()

    def load_bert_param(self, pretraining=False):
        filename = self.url.split('/')[-1]
        config = "{}/{}".format(filename.split('.')[0], "bert_config.json")
        vocab_file = "{}/{}".format(filename.split('.')[0], "vocab.txt")
        model_path = "{}/{}".format(filename.split('.')[0], "bert_model.ckpt")
        bert_param = json.load(open(config, 'r'))
        if not pretraining and self.lg == 'zh':
            bert_param.pop("directionality")
            bert_param.pop("pooler_fc_size")
            bert_param.pop("pooler_num_attention_heads")
            bert_param.pop("pooler_num_fc_layers")
            bert_param.pop("pooler_size_per_head")
            bert_param.pop("pooler_type")
        if not pretraining and self.lg == 'en':
            pass
        bert_param["batch_size"] = 32
        bert_param["maxlen"] = 80
        bert_param["label_size"] = 10
        return bert_param, vocab_file, model_path
