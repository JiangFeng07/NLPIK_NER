#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/22 11:39
# @Author: lionel
import json
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
