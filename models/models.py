#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/11 13:58
# @Author: lionel
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
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


class SinusoidalPositionalEmbedding(object):
    def __init__(self, d_model, max_len=512):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def get_embedding(self):
        pe = torch.zeros(size=(self.max_len, self.d_model))
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, self.d_model, 2) * (math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class AbsoluteSinusoidalPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(AbsoluteSinusoidalPositionalEncoder, self).__init__()
        pe = SinusoidalPositionalEmbedding(d_model, max_len).get_embedding()
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x (batch_size, seq_len, embed_size)
        pe = self.pe.unsqueeze(0)[:, :x.size(1)]
        x = x + pe
        return x


class RelativeSinusoidalPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_pos, max_len=512):
        super(RelativeSinusoidalPositionalEncoder, self).__init__()
        pe = SinusoidalPositionalEmbedding(d_model, max_len).get_embedding()
        self.register_buffer('pe', pe)
        self.max_pos = max_pos
        self.d_model = d_model

    def get_position_matrix(self, seq_len):
        positions = np.zeros((seq_len, seq_len, self.d_model))
        for i in range(seq_len):
            for j in range(seq_len):
                if j < i:
                    positions[i][j] = self.pe[max(0, self.max_pos - (i - j)), :]
                elif j > i:
                    positions[i][j] = self.pe[min(2 * self.max_pos, self.max_pos + (j - i)), :]
                else:
                    positions[i][j] = self.pe[self.max_pos, :]
        return positions

    def forward(self, x):
        batch_size, seq_len = x.size()
        position_list = np.zeros((batch_size, seq_len, seq_len, self.d_model))
        for batch_idx in range(batch_size):
            position_list[batch_idx] = self.get_position_matrix(seq_len)
        return torch.tensor(position_list)


class DotProductAttention(object):
    """ 乘法注意力 """

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


class MultiHeadedAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, heads, d_model):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.attention = DotProductAttention()
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsquezee(1)
        batch_size = query.size(0)

        # query：(batch_size, seq_len, heads, d_k), key, value same
        query, key, value = [linear(x).view(batch_size, -1, self.heads, self.d_k) for linear, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = self.attention.attention(query, key, value, mask)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        return self.linears[-1](x)


if __name__ == '__main__':
    a = torch.randint(0, 10, size=(4, 5, 512), dtype=torch.float)
    multi_head = MultiHeadedAttention(heads=8, d_model=512)
    b = multi_head(a, a, a)
    x = torch.randn(size=(3, 6))
    # pe = AbsoluteSinusoidalPositionalEncoder(8, 100)

    pe = RelativeSinusoidalPositionalEncoder(d_model=100, max_pos=4)
    # d = pe.get_position_matrix(10)
    c = pe(x)
    print(c.size())
    # print(torch.tensor(d).detach())
