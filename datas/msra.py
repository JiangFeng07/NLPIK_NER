#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/22 11:39
# @Author: lionel
import json

import pandas as pd
from torch.utils import data


class MsraNerDataset(data.Dataset):
    def __init__(self, file_path, max_len):
        super(MsraNerDataset, self).__init__()
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip('\n'), encoding='utf-8'))
        self.max_len = max_len

    def __getitem__(self, item):
        if isinstance(self.data[item], dict):
            chars = self.data[item].get('chars')
            labels = self.data[item].get('labels')
            char_len = len(chars)
            if len(chars) > self.max_len:
                chars = chars[:self.max_len]
                labels = labels[:self.max_len]
                char_len = len(chars)

            return chars, labels, char_len

    def __len__(self):
        return len(self.data)


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


if __name__ == '__main__':
    data = MsraNerDataset(file_path='/tmp/MSRA_NER/train.json', max_len=100)
    for ele in data:
        print(ele)
        break
