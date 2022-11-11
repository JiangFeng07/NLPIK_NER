#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/9/29 17:47
# @Author: lionel
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from models.models import BilstmEncoder


class BILSTM_CRF(nn.Module):
    """ 实体识别模型 BILstm_Crf 代码实现"""

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout):
        super(BILSTM_CRF, self).__init__()
        self.encoder = BilstmEncoder(vocab_size, embedding_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, x, seq_len, y=None, mask=None):
        outputs = self.encoder(x, seq_len)
        outputs = self.fc(outputs)
        if y is not None:
            outputs = self.dropout(outputs)
        logits = self.softmax(outputs)

        if y is None:
            pred = self.crf.decode(logits, mask)
            return pred
        else:
            loss = -1 * self.crf(logits, y, mask)
            return loss
