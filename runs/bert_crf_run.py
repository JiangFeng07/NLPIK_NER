#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/3 13:53
# @Author: lionel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bert_model_path', help='中文bert预训练模型路径', type=str, default='/tmp/chinese-roberta-wwm-ext')
    parser.add_argument('file_path', help='模型训练数据路径', type=str, default='/tmp/msra')
    parser.add_argument('epochs', help='模型训练轮数', type=int, default=1)
    args = parser.parse_args()
