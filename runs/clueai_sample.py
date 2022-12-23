#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/12/16 13:21
# @Author: lionel
# import clueai
#
# cl = clueai.Client('', check_api_key=False)
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("/tmp/PromptCLUE-base")
model = AutoModelForSeq2SeqLM.from_pretrained("/tmp/PromptCLUE-base")
model.to(device)


def get_ner_prompt(text):
    prompt = """
        信息抽取：
        %s
        问题：机构名，人名，地址，时间，职位
        答案：
        """.format(text)
    return prompt


def preprocess(text):
    return text.replace("\n", "_")


def postprocess(text):
    return text.replace("_", "\n")


def answer(text, sample=False, top_p=0.6):
    '''sample：是否抽样。生成任务，可以设置为True;
       top_p：0-1之间，生成的内容越多样、
    '''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
    if not sample:  # 不进行采样
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4,
                             length_penalty=0.6)
    else:  # 采样（生成）
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128,
                             do_sample=True, top_p=top_p)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


if __name__ == '__main__':
    text = ''
    start = time.time()
    print(answer(get_ner_prompt(text)))
    print(time.time() - start)
