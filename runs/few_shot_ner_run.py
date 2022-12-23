#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/30 00:04
# @Author: lionel
import argparse
import os.path

import torch
from torch.utils import data
from tqdm import tqdm
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

from datas.msra import MsraNerDataset
from models.few_shot_ner import LabelSemanticsFSNER
from models.utils import tokenizer, build_vocab, entity_decode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch, vocab2id, label2id):
    chars, labels, _ = zip(*batch)
    token_ids, _, attention_mask = tokenizer(chars, vocab2id)
    batch_size, seq_len = token_ids.size()
    gold_labels = torch.zeros(size=(batch_size, seq_len), dtype=torch.long, device=device)
    for i, label in enumerate(labels):
        for j, ele in enumerate(label):
            gold_labels[i][j] = label2id[ele]
    return token_ids, attention_mask, gold_labels, chars, labels


def metric(dataloader, model, id2label, epoch):
    correct_num, predict_num, gold_num = 0, 0, 0
    with tqdm(total=len(dataloader), desc='Epoch：%d，模型训练进度条' % (epoch + 1)) as pbar:
        for batch_idx, batch in enumerate(dataloader):
            token_ids, token_attention_mask, _, chars, labels = batch
            logits = model(token_ids, token_attention_mask)
            pred_labels = torch.argmax(logits, dim=-1)
            for char, pred_label, label in zip(chars, pred_labels, labels):
                pred_label = [id2label[int(_label)] for _label in pred_label[:len(char)]]

                pred_entity_set = set()
                for key, val in entity_decode(char, pred_label).items():
                    pred_entity_set.add((key, val['value']))
                gold_entity_set = set()
                for key, val in entity_decode(char, label).items():
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


def train():
    vocab2id, id2vocab = build_vocab(os.path.join(args.bert_model_path, 'vocab.txt'))
    label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-COM": 3, "I-COM": 4, "B-LOC": 5, "I-LOC": 6, "B-ORG": 7, "I-ORG": 8}
    id2label = {val: key for key, val in label2id.items()}
    train_dataset = MsraNerDataset(os.path.join(args.file_path, 'train.json'), max_len=128)
    train_dataloader = data.DataLoader(train_dataset, batch_size=8, shuffle=True,
                                       collate_fn=lambda ele: collate_fn(ele, vocab2id, label2id))
    valid_dataset = MsraNerDataset(os.path.join(args.file_path, 'dev.json'), max_len=128)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=8,
                                       collate_fn=lambda ele: collate_fn(ele, vocab2id, label2id))
    labels = ['人名开始', '人名中间', '公司名开始', '公司名中间', '地址名开始', '地址名中间', '机关单位名开始', "机关单位名中间", '其它']
    label_token_ids, _, label_attention_mask = tokenizer(labels, vocab2id)

    label_encoder = BertModel.from_pretrained(args.bert_model_path)
    text_encoder = BertModel.from_pretrained(args.bert_model_path)
    model = LabelSemanticsFSNER(label_encoder, text_encoder, label_token_ids, label_attention_mask)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)
    best_f1_score = 0.0
    early_epochs = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len(train_dataloader), desc='Epoch：%d，模型训练进度条' % (epoch + 1)) as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                token_ids, token_attention_mask, gold_labels, _, _ = batch
                optimizer.zero_grad()
                logits = model(token_ids, token_attention_mask)
                gold_labels = token_attention_mask * gold_labels
                loss = loss_fn(logits.view((-1, logits.size(-1))), gold_labels.view(-1))
                pbar.set_postfix({'loss': '{0:1.5f}'.format(float(loss))})
                pbar.update()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                break
        with torch.no_grad():
            model.eval()
            precision, recall, f1_score = metric(valid_dataloader, model, id2label, epoch)
            if f1_score > best_f1_score:
                torch.save(model.state_dict(), args.model_path)
                best_f1_score = f1_score
                early_epochs = 0
            else:
                early_epochs += 1

            if early_epochs > 3:  # 连续三个epoch，验证集f1_score没有提升，训练结束
                print('验证集f1_score连续三个epoch没有提升，训练结束')
                break
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model_path', help='中文bert预训练模型路径', type=str, default='/tmp/chinese-roberta-wwm-ext')
    parser.add_argument('--file_path', help='模型训练数据路径', type=str, default='/tmp/MSRA_NER')
    parser.add_argument('--epochs', help='模型训练轮数', type=int, default=1)
    parser.add_argument('--batch_size', help='', type=int, default=32)
    parser.add_argument('--warm_up_ratio', help='模型训练轮数', type=float, default=0.1)
    parser.add_argument('--lr', help='学习率', type=float, default=2e-5)
    parser.add_argument('--model_path', help='模型存储路径', type=str, default='')
    args = parser.parse_args()
    train()
