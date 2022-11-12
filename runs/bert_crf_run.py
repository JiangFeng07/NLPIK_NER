#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/3 13:53
# @Author: lionel
import argparse
import json
import os

import torch
from torch.utils import data
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel

from datas.msra import MsraNerDataset
from datas.utils import build_vocab
from models.bert_crf import Bert_CRF
from models.utils import tokenizer, metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch, vocab2id, label2id):
    texts, labels, char_lens = zip(*batch)
    token_ids, token_type_ids, attention_mask = tokenizer(texts, vocab2id, device)
    batch_size, seq_len = token_ids.size()

    gold_labels = torch.zeros(size=(batch_size, seq_len), dtype=torch.int32, device=device)

    for i, label in enumerate(labels):
        for j, ele in enumerate(label):
            gold_labels[i][j] = label2id[ele]

    return token_ids, token_type_ids, attention_mask, gold_labels, texts, labels, char_lens


def train():
    label2id = json.load(open(os.path.join(args.file_path, 'tag.json'), 'r', encoding='utf-8'))
    id2label = {val: key for key, val in label2id.items()}
    vocab2id, _ = build_vocab(os.path.join(args.file_path, 'vocab.txt'))
    num_labels = len(label2id)
    train_data = MsraNerDataset(file_path=os.path.join(args.file_path, 'train.json'), max_len=100)
    valid_data = MsraNerDataset(file_path=os.path.join(args.file_path, 'dev.json'), max_len=100)
    train_loader = data.DataLoader(train_data, batch_size=16, shuffle=True,
                                   collate_fn=lambda ele: collate_fn(ele, vocab2id, label2id))
    valid_loader = data.DataLoader(valid_data, batch_size=10,
                                   collate_fn=lambda ele: collate_fn(ele, vocab2id, label2id))
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    model = Bert_CRF(encoder=bert_model, num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_f1_score = 0.0
    early_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len(train_loader), desc='模型训练进度条') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                input_ids, token_type_ids, attention_mask, labels, _, _, _ = batch
                optimizer.zero_grad()
                loss = model(input_ids, token_type_ids, attention_mask, labels)
                pbar.set_postfix({'loss': '{0:1.5f}'.format(float(loss))})
                pbar.update()
                loss.backward()
                optimizer.step()
                scheduler.step()

        model.eval()
        with torch.no_grad():
            precision, recall, f1_score = metric(valid_loader, model, id2label)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model_path', help='中文bert预训练模型路径', type=str, default='/tmp/chinese-roberta-wwm-ext')
    parser.add_argument('--file_path', help='模型训练数据路径', type=str, default='/tmp/')
    parser.add_argument('--epochs', help='模型训练轮数', type=int, default=1)
    parser.add_argument('--model_path', help='模型存储路径', type=str, default='')
    args = parser.parse_args()
    train()
