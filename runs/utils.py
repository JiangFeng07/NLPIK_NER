#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/10 17:01
# @Author: lionel
import json

import pandas as pd
import torch
from transformers import BertModel


# 参考文档 https://wmathor.com/index.php/archives/1537/

class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        for name, param in bert_model.named_parameters():
            if param.required_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认2范数, g/|g|
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        for name, param in bert_model.named_parameters():
            if param.required_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


"""
# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    # 对抗训练
    fgm.attack() # embedding被修改了
    # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
    loss_sum = model(batch_input, batch_label)
    loss_sum.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
    fgm.restore() # 恢复Embedding的参数
    # 梯度下降，更新参数
    optimizer.step()
    optimizer.zero_grad()
"""


class PGD(object):
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.required_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.required_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


"""
pgd = PGD(model)
K = 3
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    pgd.backup_grad() # 保存正常的grad
    # 对抗训练
    for t in range(K):
        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != K-1:
            optimizer.zero_grad()
        else:
            pgd.restore_grad() # 恢复正常的grad
        loss_sum = model(batch_input, batch_label)
        loss_sum.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    pgd.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    optimizer.zero_grad()
"""


def tokenizer(texts, vocab, device):
    """
        bert model input
    :param texts: 文本集合
    :param vocab: 词典
    :return:
    """
    batch_size = len(texts)
    seq_len = max([len(text) for text in texts]) + 2

    token_ids = torch.zeros((batch_size, seq_len), dtype=torch.int, device=device)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.int, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.int, device=device)

    for index, text in enumerate(texts):
        token_ids[index][0] = vocab['[CLS]']
        attention_mask[index][0] = 1
        i = 0
        while i < len(text):
            token_ids[index][i + 1] = vocab.get(text[i], vocab['[UNK]'])
            attention_mask[index][i + 1] = 1
            i += 1
        token_ids[index][i + 1] = vocab['[SEP]']
        attention_mask[index][i + 1] = 1
    return token_ids, token_type_ids, attention_mask


def entity_decode(chars, labels, mode='BIO'):
    """
        序列标注实体解析
    :param chars: 字符集合
    :param labels: 字符对应标签集合
    :param mode: 标注方式：BIOES 和 BIO（默认）
    :return: 实体集合
    """

    entities = {}
    assert len(chars) == len(labels)
    i, start = 0, len(chars)
    if mode == 'BIOES':
        while i < len(labels):
            if labels[i] == 'O':
                i += 1
                continue
            flag, label = labels[i].split('-')
            if flag == 'S':
                entities[chars[i]] = label
            elif flag == 'B':
                start = i
            elif flag == 'E':
                entities[''.join(chars[start:i + 1])] = label
            i += 1
        return entities
    elif mode == 'BIO':
        tag = ''
        while i < len(labels):
            if 'B-' in labels[i]:
                tag = labels[i][2:]
                start = i
            elif 'I-' in labels[i]:
                while i + 1 < len(labels) and 'I-' in labels[i + 1]:
                    i += 1
                entities[''.join(chars[start:i + 1])] = tag
            i += 1
        return entities

    print('暂不支持该中标注方式解码')
    return entities





if __name__ == '__main__':
    chars = ["桐", "乡", "市", "濮", "院", "镇", "凯", "旋", "路", "0", "0", "0", "0", "弄", "0", "单", "元", "电", "联"]
    labels = ["B-district", "I-district", "E-district", "B-town", "I-town", "E-town", "B-road",
              "I-road", "E-road", "B-road", "I-road", "I-road", "I-road", "E-road", "B-cellno", "I-cellno", "E-cellno",
              "O", "O"]
    entities = entity_decode(chars, labels, mode='BIOES')
    print(entities)

    chars = ['正', '当', '朱', '镕', '基', '当', '选', '政', '府', '总', '理', '后', '第', '一', '次', '在', '中', '外', '记', '者', '招',
             '待', '会', '上', '，', '回', '答', '外', '国', '记', '者', '的', '提', '问', '：', '中', '国', '农', '村', '是', '否', '实',
             '行', '民', '主', '选', '举', '制', '度', '的', '时', '候', '，', '一', '位', '中', '国', '电', '视', '编', '导', '正', '携',
             '带', '着', '她', '的', '反', '映', '中', '国', '农', '村', '民', '主', '选', '举', '村', '委', '会', '领', '导', '的', '电',
             '视', '纪', '录', '片', '《', '村', '民', '的', '选', '择', '》', '（', '北', '京', '电', '视', '台', '摄', '制', '，', '仝',
             '丽', '编', '导', '）', '，', '出', '现', '在', '法', '国', '的', '真', '实', '电', '影', '节', '上', '。']
    labels = ['O', 'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O',
              'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC',
              'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC',
              'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O',
              'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
              'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O',
              'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    entities = entity_decode(chars, labels)
    print(entities)
