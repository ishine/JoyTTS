#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import numpy as np
import torchaudio
import whisper

import os
import time
import json
from transformers import AutoModel, AutoTokenizer
# from minicpm.modle import MiniCPMO

class AgentChat():
    
    def __init__(self, model_path, device=None):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        model = model.eval()
        if device is not None:
            model = model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = model
         

    def generate(self, target_text_list):
        user_question = [[{'role': 'user', 'content': [f"请重复下面的文本.",target_text]}] for target_text in target_text_list]
            
        # self.msgs_history.append(user_question)
        msgs_history=user_question
        # import ipdb; ipdb.set_trace()
        text_unpad, text_token, text_token_len, hidden_states = self.model.get_hidden(
            msgs=msgs_history,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
            chunk_input=False,
            output_audio_path=None,
            # answer=[input],
        )
        return text_unpad, text_token, text_token_len, hidden_states
    

    def forward(self, target_text_list):
        user_question = [[{'role': 'user', 'content': [f"请重复下面的文本.",target_text]}] for target_text in target_text_list]
            
        # self.msgs_history.append(user_question)
        msgs_history=user_question
        # import ipdb; ipdb.set_trace()
        text_unpad, text_token, text_token_len, hidden_states = self.model.get_hidden(
            msgs=msgs_history,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
            chunk_input=False,
            output_audio_path=None,
            # answer=[input],
        )
        return text_unpad, text_token, text_token_len, hidden_states


def single_job(utt):
    target_text = utt2text[utt]
    chatsystem(target_text)
    return utt, speech_token


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2speech_token = {}
    for future in tqdm(as_completed(all_task)):
        utt, speech_token = future.result()
        utt2speech_token[utt] = speech_token
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dir", type=str)
#     parser.add_argument("--cpm_path", type=str)
#     parser.add_argument("--num_thread", type=int, default=8)
#     args = parser.parse_args()

#     utt2text = {}
#     with open('{}/text'.format(args.dir)) as f:
#         for l in f:
#             l = l.replace('\n', '').split()
#             utt2text[l[0]] = ' '.join(l[1:])

#     chatsystem = AgentChat(args.cpm_path)

#     executor = ThreadPoolExecutor(max_workers=args.num_thread)

#     main(args)

if __name__ == "__main__":
    chatsystem = AgentChat("/mnt/afs/zhoufangru/agent/end2end/pretrained_models/MiniCPM-o-2_6")
    target_text_list = ['     与  六月  多达  百分  之  五  的  改善  型  购房  占  比  相比 ',
                        '香港 演艺圈 欢迎 毛阿敏 加盟 无线 台 与 华星 一些 重大 的 演唱 活动 都 邀请 她 出场 有几次 还 特意 安排 压轴 演出',
                        ]
    for i in range(10):
        chatsystem.generate(target_text_list)