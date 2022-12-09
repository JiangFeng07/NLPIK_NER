#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/3 13:32
# @Author: lionel
import json
import os.path

from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from tqdm import tqdm
from datas.msra import MsraNerDataset
from models.bilstm_crf import BILSTM_CRF
import torch

from models.utils import metric, build_vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch, vocab2id, label2id):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    chars, labels, char_len = zip(*batch)
    features, label_list, seq_len = [], [], []
    for char, label in zip(chars, labels):
        features.append(torch.LongTensor([vocab2id.get(ele, vocab2id.get('[UNK]')) for ele in char]))
        label_list.append(torch.LongTensor([label2id[ele] for ele in label]))
    reviews = pad_sequence(features, batch_first=True).to(device)
    size = reviews.shape[1]
    masks = []
    for length in char_len:
        mask = [True] * length + [False] * (size - length)
        masks.append(mask)
    masks = torch.ByteTensor(masks, device=device)
    label_list = pad_sequence(label_list, batch_first=True).to(device)
    seq_len = torch.LongTensor(char_len, device=device)
    return label_list, reviews, seq_len, masks


def train():
    """实体识别模型——BILstm+crf训练"""
    train_dataset = MsraNerDataset(os.path.join(args.file_path, 'train.json'), max_len=500)
    valid_dataset = MsraNerDataset(os.path.join(args.file_path, 'dev.json'), max_len=500)
    vocab2id, id2vocab = build_vocab(os.path.join(args.file_path, 'vocab.txt'))
    label2id = json.load(open(os.path.join(args.file_path, 'tag.json'), 'r', encoding='utf-8'))
    id2label = {val: key for key, val in label2id.items()}
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                   collate_fn=lambda ele: collate_fn(ele, vocab2id, label2id))
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                   collate_fn=lambda ele: collate_fn(ele, vocab2id, label2id))

    model = BILSTM_CRF(vocab_size=len(vocab2id), embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                       num_layers=args.num_layers, num_classes=len(label2id), dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len(train_loader), desc='模型训练进度条') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                label_list, reviews, seq_len, masks = batch
                optimizer.zero_grad()
                loss = model(reviews, seq_len, label_list, mask=masks)
                pbar.set_postfix({'loss': '{0:1.5f}'.format(float(loss))})
                pbar.update()
                loss.backward()
                optimizer.step()
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
