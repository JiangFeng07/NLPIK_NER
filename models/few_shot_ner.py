#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/29 23:02
# @Author: lionel
import torch
from torch import nn
from transformers import BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LabelSemanticsFSNER(nn.Module):
    def __init__(self, base_model_path, tokenizer, tag2id, label_context):
        super(LabelSemanticsFSNER, self).__init__()
        self.label_encoder = BertModel.from_pretrained(base_model_path).to(device)
        self.token_encoder = BertModel.from_pretrained(base_model_path).to(device)
        self.tag2id = tag2id
        self.index_context = {'B': '开始词', 'I': '中间词'}
        self.label_context = label_context
        self.tokenizer = tokenizer

    def build_label_representation(self, tag2id):
        labels = []
        for k, v in tag2id.items():
            if k != 'O':
                idx, label = k.split('-')
                label = self.label_context[label]
                labels.append(label + self.index_context[idx])
            else:
                labels.append('其它类别词')
        inputs_ids = self.tokenizer(labels)
        label_embeddings = self.label_encoder(inputs_ids=inputs_ids['input_ids'].to(device),
                                              token_type_ids=inputs_ids['token_type_ids'].to(device),
                                              attention_mask=inputs_ids['attention_mask'].to(device)).pooler_output
        return label_embeddings

    def forward(self, token_ids, attention_mask):
        self.label_representation = self.build_label_representation(self.tag2id).to(device).detach()
        tag_lens, hidden_size = self.label_representation.size()
        token_outputs = self.token_encoder(token_ids,
                                           attention_mask).last_hidden_state  # (batch_size, seq_len, hidden_size)
        batch_size = token_outputs.size(0)
        label_embeddings = self.label_representation.expand(batch_size, tag_lens, hidden_size)
        logits = torch.matmul(token_outputs, torch.transpose(label_embeddings, 2, 1))
        softmax_embeddings = torch.nn.Softmax(dim=2)(logits)
        label_indexs = torch.argmax(softmax_embeddings, dim=-1)
        return logits, label_indexs
