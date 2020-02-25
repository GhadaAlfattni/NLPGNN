#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import numpy as np
class tokenizer():
    def __init__(self,item_index):
        self.item_index = item_index

    def sequence_2_index(self,sequence):
        return [[self.item_index.get(item) for item in items] for items in sequence]

    def index_2_token(self,sequence):
        self.index_item = {index:item for item,index in self.item_index.items()}
        return [[self.index_item.get(item) for item in items] for items in sequence]






