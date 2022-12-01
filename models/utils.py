#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/10 17:01
# @Author: lionel
import json
import re

import torch
from tqdm import tqdm


# 参考文档 https://wmathor.com/index.php/archives/1537/

class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认2范数, g/|g|
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
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
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
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


def entity_label_encode(chars, entities):
    """"""
    tag_list = ['O'] * len(chars)
    if not entities:
        return tag_list
    entity_list = sorted(entities.items(), key=lambda ele: -len(ele[0]))
    parsed_index = []
    for entity, tag in entity_list:
        entity_len = len(entity)
        i = 0
        while i < len(chars) - entity_len:
            if ''.join(chars[i:i + entity_len]) == entity:
                flag = True
                for _i in range(i, i + entity_len):
                    if _i in parsed_index:
                        flag = False
                if not flag:
                    i += 1
                    break
                tag_list[i] = 'B-%s' % tag
                parsed_index.append(i)
                j = i + 1
                while j < i + entity_len:
                    tag_list[j] = 'I-%s' % tag
                    parsed_index.append(j)
                    j += 1
                i += entity_len
            else:
                i += 1

    return tag_list


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
                _label = {'value': label, 'start': i, 'end': i}
                entities[chars[i]] = _label
            elif flag == 'B':
                start = i
            elif flag == 'E':
                _label = {'value': label, 'start': start, 'end': i}
                entities[''.join(chars[start:i + 1])] = _label
            i += 1
        return entities
    elif mode == 'BIO':
        label = ''
        while i < len(labels):
            if 'B-' in labels[i]:
                label = labels[i][2:]
                start = i
            elif 'I-' in labels[i]:
                while i + 1 < len(labels) and 'I-' in labels[i + 1]:
                    i += 1
                _label = {'value': label, 'start': start, 'end': i}
                entities[''.join(chars[start:i + 1])] = _label
            i += 1
        return entities

    print('暂不支持该中标注方式解码')
    return entities


def metric(dataloader, model, id2label):
    correct_num, predict_num, gold_num = 0, 0, 0
    with tqdm(total=len(dataloader), desc='模型验证进度条') as pbar:
        for index, batch in enumerate(dataloader):
            token_ids, token_type_ids, attention_mask, _, texts, gold_labels, char_lens = batch
            pred_labels = model(token_ids, token_type_ids, attention_mask, None)

            for chars, pred_label, gold_label, char_len in zip(texts, pred_labels, gold_labels, char_lens):
                pred_label = [id2label[_label] for _label in pred_label[:char_len]]
                pred_entities = entity_decode(chars, pred_label, mode='BIOES')
                gold_entities = entity_decode(chars, gold_label, mode='BIOES')
                pred_entity_set = set()
                gold_entity_set = set()
                for key, val in pred_entities.items():
                    pred_entity_set.add((key, val['value']))
                for key, val in gold_entities.items():
                    gold_entity_set.add((key, val['value']))
                predict_num += len(pred_entity_set)
                gold_num += len(gold_entity_set)
                correct_num += len(pred_entity_set & gold_entity_set)
            pbar.update()

    print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    print('f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}'.format(f1_score, precision, recall))
    return precision, recall, f1_score


def text_to_chars(text):
    """文本转化为字符数组"""
    char_list = []
    if not text:
        return char_list
    text = text.lower()
    if len(text) == 1:
        char_list = list(text)
        return char_list
    start = 0
    pattern = re.compile('^[\u4e00-\u9fa5， ,；。;:：]$')
    while start < len(text) - 1:
        char = text[start]
        if pattern.search(char):
            char_list.append(char)
            start += 1
            continue
        end = start + 1
        while end < len(text):
            if not pattern.search(text[end]):
                end += 1
                continue
            break
        char_list.append(text[start:end])
        start = end
    if start == len(text) - 1:
        char_list.append(text[start])
    return char_list


if __name__ == '__main__':
    chars = ["桐", "乡", "市", "濮", "院", "镇", "凯", "旋", "路", "0", "0", "0", "0", "弄", "0", "单", "元", "电", "联"]
    labels = ["B-district", "I-district", "E-district", "B-town", "I-town", "E-town", "B-road",
              "I-road", "E-road", "B-road", "I-road", "I-road", "I-road", "E-road", "B-cellno", "I-cellno", "E-cellno",
              "O", "O"]
    # entities = entity_decode(chars, labels, mode='BIOES')
    # print(entities)

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
    # entities = entity_decode(chars, labels)

    chars = ['因', '有', '关', '日', '寇', '在', '京', '掠', '夺', '文', '物', '详', '情', '，', '藏', '界', '较', '为', '重', '视', '，',
             '也', '是', '我', '们', '收', '藏', '苏', '北', '京', '饭', '店', '史', '料', '中', '的', '要', '件', '之', '一', '。']
    entities = {'苏北': 'LOC', '北京饭店': 'LOC'}
    print(entity_label_encode(chars, entities))

