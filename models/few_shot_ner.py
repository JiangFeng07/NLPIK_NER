#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/29 23:02
# @Author: lionel
import torch
from torch import nn
from transformers import BertTokenizer, BertModel


class LabelSemanticsFSNER(nn.Module):
    def __init__(self, label_encoder, token_encoder):
        super(LabelSemanticsFSNER, self).__init__()
        self.label_encoder = label_encoder
        self.token_encoder = token_encoder

    def forward(self, token_ids, token_attention_mask, label_ids, label_attention_mask):
        token_outputs = self.token_encoder(token_ids, token_attention_mask)[0]  # (batch_size, seq_len, hidden_size)
        label_outputs = self.label_encoder(label_ids, label_attention_mask)[0]
        label_cls_output = label_outputs[:, 0, :]  # (num_label, hidden_size)
        logits = torch.matmul(token_outputs, torch.transpose(label_cls_output, 1, 0))
        return torch.nn.Softmax(dim=2)(logits)


if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    bert_model = BertModel.from_pretrained(bert_model_path)
    labels = ['人名开始', '人名中间', '公司名开始', '公司名中间', '其它']
    texts = ['李四是华为信息科技有限公司的员工', '张三曾在美团点评任职']
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    encoded_outputs = tokenizer(texts, return_tensors='pt', padding=True)
    token_ids, token_attention_mask = encoded_outputs['input_ids'], encoded_outputs['attention_mask']
    label_outputs = tokenizer(labels, return_tensors='pt', padding=True)
    label_ids, label_attention_mask = label_outputs['input_ids'], label_outputs['attention_mask']
    fs_ner = LabelSemanticsFSNER(bert_model, bert_model)
    logits = fs_ner(token_ids, token_attention_mask, label_ids, label_attention_mask)
    print(logits.size())
    print(logits)
