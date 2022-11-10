#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/10 17:01
# @Author: lionel
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
        for name, param in bert_model.named_parameters():
            if param.required_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        for name, param in bert_model.named_parameters():
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
if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    bert_model = BertModel.from_pretrained(bert_model_path)
