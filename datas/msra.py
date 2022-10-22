#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/22 11:39
# @Author: lionel
import pandas as pd
from torch.utils import data


class MsraNerDataset(data.Dataset):
    def __init__(self, file_path, max_len):
        super(MsraNerDataset, self).__init__()
        self.data = pd.read_json(file_path)
        self.max_len = max_len
        self.label_dict = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-PER': 5, 'I-PER': 6}
        self.id2label = {val: key for key, val in self.label_dict.items()}

    def __getitem__(self, item):
        chars = self.data.iloc[item, 0]
        labels = self.data.iloc[item, 1]
        char_len = self.data.iloc[item, 2]

        if len(chars) > self.max_len:
            chars = chars[:self.max_len]
            labels = labels[:self.max_len]
            char_len = self.max_len
        return chars, labels, char_len

    def __len__(self):
        return self.data.shape[0]

class ClueNerDataset(data.Dataset):
    def __init__(self, file_path):
        super(ClueNerDataset, self).__init__()
        self.data = pd.read_json(file_path)
        self.labels = {'address': 0, 'book': 1, 'company': 2, 'game': 3, 'government': 4,
                       'movie': 5, 'name': 6, 'organization': 7, 'position': 8, 'scene': 9}

    def __getitem__(self, item):
        text = self.data.iloc[item, 0]
        label = self.data.iloc[item, 1]
        label_dict = dict()
        for key, val in label.items():
            for indexs in val.values():
                for index in indexs:
                    if key not in label_dict.keys():
                        label_dict[key] = []
                    label_dict[key].append(index)
        return text, label_dict

    def __len__(self):
        return self.data.shape[0]