#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/11 13:58
# @Author: lionel
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BilstmEncoder(nn.Module):
    """ 双向lstm编码"""

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
    """预训练模型编码"""

    def __init__(self, encoder):
        super(PreTrainModelEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = encoder.config.hidden_size

    def forward(self, token_ids, token_type_ids, attention_mask):
        outputs = self.encoder(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs


class AbsoluteSinusoidalPositionalEmbedding(object):
    def __init__(self, d_model, max_len=5000):
        super(AbsoluteSinusoidalPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def get_embedding(self):
        pe = torch.zeros(size=(self.max_len, self.d_model))
        position = torch.arange(0, self.max_len)
        div_term = torch.exp(-torch.arange(0, self.d_model, 2) * (math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class DotProductAttention(object):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def attention(self, Q, K, V, mask=None, scaled=True):
        """
        :param Q: (batch_size, seq_len, d_q)
        :param K: (batch_size, seq_len, d_k)
        :param V: (batch_size, seq_len, d_v)
        :param mask:
        :param scaled:
        :return:
        """
        scores = torch.matmul(Q, torch.transpose(K, -2, -1))
        if scaled:
            d_k = K.size(-1)
            scores /= math.sqrt(d_k)
        if mask:
            scores = torch.masked_fill(scores, mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        return torch.matmul(p_attn, V), p_attn


if __name__ == '__main__':
    a = torch.randint(0, 10, size=(3, 4, 5))
    print(a.size())
    print(a.T.size())
    print(torch.transpose(a, -2, -1).size())
    print(-1e9)
