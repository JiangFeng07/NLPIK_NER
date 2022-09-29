#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/9/29 17:47
# @Author: lionel
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class BILSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout):
        super(BILSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, x, seq_len, y=None, mask=None):
        x = self.embedding(x)
        x = pack_padded_sequence(x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.bilstm(x)
        outputs = pad_packed_sequence(outputs, batch_first=True)[0]
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


if __name__ == '__main__':
    bilstm_crf = BILSTM_CRF(10000, 200, 100, 2, 8, 0.2)
    print(bilstm_crf)
