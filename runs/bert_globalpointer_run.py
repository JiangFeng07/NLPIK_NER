#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/11 14:44
# @Author: lionel
import os

import torch
from torch.utils import data
from tqdm import tqdm
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

from datas.msra import MsraNerDataset
from datas.utils import build_vocab
from models.bert_globalpointer import Bert_GlobalPointer, GlobalPointerLoss
from runs.utils import tokenizer, entity_decode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch, vocab2id, tags):
    texts, labels, char_lens = zip(*batch)
    token_ids, token_type_ids, attention_mask = tokenizer(texts, vocab2id, device)
    batch_size, seq_len = token_ids.size()
    gold_labels = torch.zeros(size=(batch_size, len(tags), seq_len, seq_len))

    for index, ele in enumerate(zip(texts, labels)):
        chars, _labels = ele
        entities = entity_decode(chars, _labels, mode='BIOES')
        for val in entities.values():
            label, start, end = val['value'], val['start'], val['end']
            gold_labels[index][tags[label]][start][end] = 1
    return token_ids, token_type_ids, attention_mask, gold_labels, texts, labels, char_lens


def metric(dataloader, model):
    correct_num, predict_num, gold_num = 0, 0, 0
    with tqdm(total=len(dataloader), desc='模型验证进度条') as pbar:
        for index, batch in enumerate(dataloader):
            input_ids, token_type_ids, attention_mask, gold_labels, _, _, _ = batch
            pred_labels = model(input_ids, token_type_ids, attention_mask)
            pred_labels = torch.gt(pred_labels, 0).int()
            for pred_label, gold_label in zip(pred_labels, gold_labels):
                gold_num += torch.sum(gold_label)
                predict_num += torch.sum(pred_label)
                correct_num += torch.sum(pred_label * gold_label)
    print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    print('f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}'.format(f1_score, precision, recall))
    return precision, recall, f1_score


def train():
    # label2id = json.load(open(os.path.join(args.file_path, 'tag.json'), 'r', encoding='utf-8'))
    # id2label = {val: key for key, val in label2id.items()}
    tags = {'houseno': 0, 'community': 1, 'town': 2, 'poi': 3, 'cellno': 4, 'distance': 5, 'assist': 6, 'prov': 7,
            'district': 8, 'road': 9, 'devzone': 10, 'subpoi': 11, 'roadno': 12, 'floorno': 13, 'city': 14,
            'village_group': 15, 'intersection': 16}

    vocab2id, _ = build_vocab(os.path.join(args.file_path, 'vocab.txt'))
    heads = len(tags)
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    model = Bert_GlobalPointer(encoder=bert_model, heads=heads, head_size=100, device=device).to(device)

    train_data = MsraNerDataset(file_path=os.path.join(args.file_path, 'train.json'), max_len=100)

    valid_data = MsraNerDataset(file_path=os.path.join(args.file_path, 'dev.json'), max_len=100)

    train_loader = data.DataLoader(train_data, batch_size=16, shuffle=True,
                                   collate_fn=lambda ele: collate_fn(ele, vocab2id, tags))
    valid_loader = data.DataLoader(valid_data, batch_size=10,
                                   collate_fn=lambda ele: collate_fn(ele, vocab2id, tags))

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_f1_score = 0.0
    early_epochs = 0
    globalPointerLoss = GlobalPointerLoss()
    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len(train_loader), desc='Epoch:{0}，模型训练进度条'.format(epoch + 1)) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                input_ids, token_type_ids, attention_mask, gold_labels, _, _, _ = batch
                optimizer.zero_grad()
                logits = model(input_ids, token_type_ids, attention_mask)
                loss = globalPointerLoss(logits, gold_labels)
                pbar.set_postfix({'loss': '{0:1.5f}'.format(float(loss))})
                pbar.update()
                loss.backward()
                optimizer.step()
                scheduler.step()

        model.eval()
        with torch.no_grad():
            precision, recall, f1_score = metric(valid_loader, model)
            if f1_score > best_f1_score:
                torch.save(model.state_dict(), args.model_path)
                best_f1_score = f1_score
                early_epochs = 0
            else:
                early_epochs += 1

            if early_epochs > 3:  # 连续三个epoch，验证集f1_score没有提升，训练结束
                print('验证集f1_score连续三个epoch没有提升，训练结束')
                break
        print('\n')
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='训练数据路径', type=str, default='/tmp/')
    parser.add_argument('--epochs', help='训练轮数', type=int, default=1)
    parser.add_argument('--dropout', help='', type=float, default=0.5)
    parser.add_argument('--embedding_size', help='', type=int, default=100)
    parser.add_argument('--batch_size', help='', type=int, default=100)
    parser.add_argument('--hidden_size', help='', type=int, default=200)
    parser.add_argument('--num_layers', help='', type=int, default=1)
    parser.add_argument('--lr', help='学习率', type=float, default=1e-3)
    args = parser.parse_args()

    train()
