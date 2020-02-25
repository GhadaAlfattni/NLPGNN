import numpy as np
import tensorflow as tf
import os
from fennlp.tokenizers import tokenization
import collections
import codecs
import pickle


class ZHTFWriter(object):
    def __init__(self, maxlen, vocab_files, modes, do_low_case=True, check_exist=False):
        self.language = "zh"
        self.maxlen = maxlen
        self.fulltoknizer = tokenization.FullTokenizer(
            vocab_file=vocab_files, do_lower_case=do_low_case
        )
        for mode in modes:
            self.mode = mode
            print("Writing {}".format(self.mode))
            self.filename = os.path.join("InputNER", self.mode)
            if check_exist:
                if os.path.exists(self.filename + ".tfrecords"):
                    self.label_map = pickle.load(open(os.path.join("InputNER", "label2id.pkl"), 'rb'))
                    print("Having Writen {} file in to device successfully!".format(mode))
                    pass
                else:
                    examples = self._read_file()
                    self.label_map = self.label2id()
                    self._write_examples(examples)
            else:
                examples = self._read_file()
                self.label_map = self.label2id()
                self._write_examples(examples)

    def _read_file(self):
        with codecs.open(self.filename, encoding='utf-8') as rf:
            examples = self._creat_examples(rf, self.mode)
        return examples

    def _creat_examples(self, lines, mode):
        examples = []
        self.label_list = set()
        for line in lines:
            line = line.strip().split('\t')
            if mode == "test":
                w = tokenization.convert_to_unicode(line[0])
                label = "0"
            else:
                w = tokenization.convert_to_unicode(line[0])
                label = tokenization.convert_to_unicode(line[-1])
            examples.append((w, label))
            self.label_list.update(set(label.split()))
        self.label_list = sorted(self.label_list)
        print("Totally use {} labels!\n".format(len(self.label_list)))
        return examples

    def _creat_features(self, values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    def label2id(self):
        label_map = {}
        for (i, label) in enumerate(self.label_list):
            label_map[label] = i
        pickle.dump(label_map, open(os.path.join("InputNER", "label2id.pkl"), 'wb'))
        return label_map

    def _convert_single_example(self, example, maxlen):
        # self.label_map =

        tokens = ["[CLS]"]
        segment_ids = [0]
        input_mask = [1]
        label_id = [self.label_map.get("O")]
        sentences, labels = example
        labels = labels.split()
        sentences = [self.fulltoknizer.tokenize(w) for w in sentences.split()]
        for i, words in enumerate(sentences):
            for word in words:
                if len(tokens) < maxlen - 1:
                    tokens.append(word)
                    segment_ids.append(0)
                    input_mask.append(1)
                    label_id.append(self.label_map.get(labels[i]))
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)
        label_id.append(self.label_map.get("O"))
        input_ids = self.fulltoknizer.convert_tokens_to_ids(tokens)
        while len(input_ids) < maxlen:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_id.append(self.label_map.get("O"))
        return input_ids, segment_ids, input_mask, label_id

    def _write_examples(self, examples):
        writer = tf.io.TFRecordWriter(os.path.join("InputNER", self.mode + ".tfrecords"))
        features = collections.OrderedDict()
        for example in examples:
            input_ids, segment_ids, input_mask, label_id = self._convert_single_example(example, self.maxlen)
            features["input_ids"] = self._creat_features(input_ids)
            features["label_id"] = self._creat_features(label_id)
            features["segment_ids"] = self._creat_features(segment_ids)
            features["input_mask"] = self._creat_features(input_mask)

            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
        writer.close()

    def convert_id_to_vocab(self, items):
        vocabs = self.fulltoknizer.convert_ids_to_tokens(items)
        return vocabs

    def convert_id_to_label(self, items):
        id2label = {value: key for key, value in self.label_map.items()}
        output = []
        for item in items:
            output.append(id2label[item])
        return output


class NERLoader(object):
    def __init__(self, maxlen, batch_size, epoch=None):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.epoch = epoch

    def decode_record(self, record):
        # 告诉解码器每一个feature的类型
        feature_description = {
            "input_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
            "label_id": tf.io.FixedLenFeature([self.maxlen], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
            "input_mask": tf.io.FixedLenFeature([self.maxlen], tf.int64),

        }
        example = tf.io.parse_single_example(record, feature_description)
        return example["input_ids"], example["segment_ids"], example["input_mask"], example["label_id"]

    def load_train(self):
        self.filename = os.path.join("InputNER", "train.tfrecords")
        raw_dataset = tf.data.TFRecordDataset(self.filename)
        dataset = raw_dataset.apply(
            tf.data.experimental.map_and_batch(
                lambda record: self.decode_record(record),
                batch_size=self.batch_size,
                drop_remainder=True))
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(
                10000,
                self.epoch
            )
        )
        return dataset

    def load_valid(self):
        self.filename = os.path.join("InputNER", "valid.tfrecords")
        raw_dataset = tf.data.TFRecordDataset(self.filename)
        dataset = raw_dataset.apply(
            tf.data.experimental.map_and_batch(
                lambda record: self.decode_record(record),
                batch_size=self.batch_size,
                drop_remainder=False))
        return dataset
