#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/11 13:58
# @Author: lionel
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BilstmEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(BilstmEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)

    def forward(self, x, seq_len):
        x = self.embedding(x)
        x = pack_padded_sequence(x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.bilstm(x)
        outputs = pad_packed_sequence(outputs, batch_first=True)[0]
        return outputs


class PreTrainModelEncoder(nn.Module):
    def __init__(self, encoder):
        super(PreTrainModelEncoder, self).__init__()
        self.encoder = encoder

    def forward(self, token_ids, token_type_ids, attention_mask):
        outputs = self.encoder(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs

