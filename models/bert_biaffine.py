#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/10 14:37
# @Author: lionel
import torch
from torch import nn
from transformers import BertModel, BertTokenizer


class BERTBiaffine(nn.Module):
    def __init__(self, encoder, num_labels, drop_out, ffnn_hidden_size):
        super(BERTBiaffine, self).__init__()
        self.encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(drop_out)
        self.head_fc = nn.Linear(self.hidden_size, ffnn_hidden_size)
        self.tail_fc = nn.Linear(self.hidden_size, ffnn_hidden_size)
        self.Um = nn.Parameter(torch.randn(size=(ffnn_hidden_size, num_labels, ffnn_hidden_size)))
        self.fc = nn.Linear(2 * ffnn_hidden_size, num_labels)

    def forward(self, token_ids, token_type_ids, attention_mask):
        sequence_output = \
            self.encoder(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
                0]  # (batch_size, seq_len, hidden_size)

        pred_heads = self.head_fc(sequence_output)  # (batch_size, seq_len, ffnn_hidden_size)
        pred_tails = self.tail_fc(sequence_output)  # (batch_size, seq_len, ffnn_hidden_size)

        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', pred_heads, self.Um,
                                       pred_tails)  # (batch_size, seq_len, seq_len, num_labels)
        outputs = self.fc(torch.concat([pred_heads, pred_tails], dim=-1))  # (batch_size, seq_len, num_labels)
        outputs = outputs.unsqueeze(1) + bilinar_mapping

        return outputs


if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    bert_model = BertModel.from_pretrained(bert_model_path)
    bert_biaffine = BERTBiaffine(encoder=bert_model, num_labels=10, drop_out=0.1, ffnn_hidden_size=150)

    print(bert_biaffine)
    texts = ['原告：张三', '被告：李四伍']
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    encoded_outputs = tokenizer(texts, return_tensors='pt', padding=True)
    token_ids, token_type_ids, attention_mask = encoded_outputs['input_ids'], encoded_outputs['token_type_ids'], \
                                                encoded_outputs['attention_mask']

    a = bert_biaffine(token_ids, token_type_ids, attention_mask)

    print(a.size())
