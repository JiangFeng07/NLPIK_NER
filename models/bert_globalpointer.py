#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/21 09:24
# @Author: lionel
import math

import torch
from torch import nn

from models.models import PreTrainModelEncoder

from models.utils import tokenizer


class Bert_GlobalPointer(nn.Module):
    """实体识别模型 Bert_GlobalPointer，参考https://kexue.fm/archives/8373"""

    def __init__(self, encoder, heads, head_size, tril_mask=True, RoPE=True, device='cpu'):
        super(Bert_GlobalPointer, self).__init__()
        self.heads = heads
        self.tril_mask = tril_mask
        self.head_size = head_size
        self.RoPE = RoPE
        self.encoder = PreTrainModelEncoder(encoder)
        self.fc1 = nn.Linear(self.encoder.hidden_size, head_size * heads * 2)
        self.device = device

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        embeddings = torch.zeros(seq_len, output_dim)
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        div_term = torch.exp(-torch.arange(0, output_dim, 2) * (math.log(10000.0) / output_dim))
        embeddings[:, 0::2] = torch.sin(position_ids * div_term)
        embeddings[:, 1::2] = torch.cos(position_ids * div_term)
        embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.repeat(batch_size, 1, 1)
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, token_ids, token_type_ids, attention_mask):
        bert_outputs = self.encoder(token_ids, token_type_ids, attention_mask)
        x = bert_outputs[0]  # [batch_size, seq_len, 768]
        batch_size, seq_len, hidden_size = x.size()
        fc = self.fc1(x)
        fc = torch.split(fc, self.head_size * 2, dim=-1)
        fc = torch.stack(fc, dim=-2)
        qw, kw = fc[..., :self.head_size], fc[..., self.head_size:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, head_size)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.head_size)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, head_size)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        return logits / self.head_size ** 0.5


class GlobalPointerLoss(nn.Module):
    def __init__(self):
        super(GlobalPointerLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: 预测标签 [batch_size, num_heads, seq_len, seq_len]
        :param y_true: 真实标签 [batch_size, num_heads, seq_len, seq_len]
        :return:
        """
        batch_size, num_heads, seq_len, seq_len = y_pred.size()

        y_true = y_true.reshape([batch_size * num_heads, seq_len ** 2])
        y_pred = y_pred.reshape([batch_size * num_heads, seq_len ** 2])
        # print(y_true)
        # print(y_pred)
        # multilabel_categorical_crossentory
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        # 填充一个0，后续通过e^0得到1
        zero_vec = torch.zeros([batch_size * num_heads, 1], device=y_pred.device)
        # zero_vec = torch.zeros([batch_size * num_heads, 1])
        y_pred_neg = torch.cat([y_pred_neg, zero_vec], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zero_vec], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return torch.mean(neg_loss + pos_loss)


class GlobalPointerServer(object):
    def __init__(self, model_path, encoder, heads, head_size, tril_mask=True, RoPE=True, device='cpu'):
        self.device = device
        self.model = Bert_GlobalPointer(encoder, heads, head_size, tril_mask, RoPE, self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def predict(self, texts, vocab2id, id2tag):
        entities = []
        token_ids, token_type_ids, attention_mask = tokenizer(texts, vocab2id, self.device)
        pred_labels = self.model(token_ids, token_type_ids, attention_mask)
        pred_labels = torch.gt(pred_labels, 0).int()
        for text, pred_label in zip(texts, pred_labels):
            entity = {}
            tag_index, idx1, idx2 = torch.where(pred_label == 1)
            for idx, tag_id in enumerate(tag_index):
                tag = id2tag[tag_id]
                start = idx1[idx]
                end = idx2[idx]
                entity[text[start:end + 1]] = tag
            entities.append(entity)
        return entities
