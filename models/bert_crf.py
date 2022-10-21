#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/20 10:51
# @Author: lionel
import torch
from torch import nn
from torchcrf import CRF
from transformers import BertModel


class Bert_CRF(nn.Module):
    def __init__(self, bert_model_path, num_classes, dropout=0.2):
        super(Bert_CRF, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        for param in self.bert_model.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask, y=None):
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        logits = self.softmax(outputs)
        mask = attention_mask == 1
        if y is None:
            pred = self.crf.decode(logits, mask)
            return pred
        else:
            loss = -1 * self.crf(logits, y, mask)
            return loss
