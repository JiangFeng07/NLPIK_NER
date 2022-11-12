#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/20 10:51
# @Author: lionel
import torch
from torch import nn
from torchcrf import CRF

from models.models import PreTrainModelEncoder
from models.utils import tokenizer, entity_decode


class Bert_CRF(nn.Module):
    """实体识别模型 Bert_Crf"""

    def __init__(self, encoder, num_labels, dropout=0.2):
        super(Bert_CRF, self).__init__()
        self.encoder = PreTrainModelEncoder(encoder)
        hidden_size = self.encoder.hidden_size
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_labels, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, token_ids, token_type_ids, attention_mask, y=None):
        outputs = self.encoder(token_ids, token_type_ids, attention_mask)[0]
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        logits = self.softmax(outputs)
        mask = attention_mask == 1
        mask = mask.type(torch.bool)
        if y is None:
            pred = self.crf.decode(logits, mask)
            return pred
        else:
            loss = -1 * self.crf(logits, y.long(), mask)
            return loss


class BertCrfServer(object):
    def __init__(self, model_path, encoder, num_labels, device='cpu'):
        self.device = device
        self.model = Bert_CRF(encoder, num_labels, dropout=0.0)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def predict(self, texts, vocab2id, id2label):
        entities = []
        token_ids, token_type_ids, attention_mask = tokenizer(texts, vocab2id, self.device)
        char_lens = [len(text) for text in texts]
        pred_labels = self.model(token_ids, token_type_ids, attention_mask)
        for chars, pred_label, char_len in zip(texts, pred_labels, char_lens):
            entity = {}
            pred_label = [id2label[_label] for _label in pred_label[:char_len]]
            pred_entities = entity_decode(chars, pred_label, mode='BIOES')
            for key, val in pred_entities.items():
                entity[key] = val['value']
            entities.append(entity)
        return entities
