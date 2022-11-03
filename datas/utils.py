#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/3 13:49
# @Author: lionel
from collections import OrderedDict


def build_vocab(vocab_file):
    """加载字典
    """
    vocab2id = OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            vocab2id[line.strip('\n')] = len(vocab2id)
    id2vocab = {val: key for key, val in vocab2id.items()}
    return vocab2id, id2vocab
