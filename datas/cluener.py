#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/22 11:41
# @Author: lionel
from torch.utils import data
import pandas as pd


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
