import torch
from PIL import Image
import librosa
import os
import math
import numpy as np
import tempfile
import soundfile as sf
import time
import json
import re
import copy

from transformers import AutoModel, AutoTokenizer
# from minicpm.modle import MiniCPMO

__dir__ = os.path.dirname(os.path.abspath(__file__))
# from load_model import load_weight
torch.manual_seed(100)


import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm



class AgentChat():
    
    def __init__(self):

        model_path = os.path.join(__dir__, f"../pretrained_models/Chat")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        model = model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.model = model

    def OutputHidden(self, input, max_new_tokens=128):
        if isinstance(input,list):
            msgs_history = input
        elif isinstance(input,dict):
            msgs_history = [input]
        elif isinstance(input,str):
            if os.path.exists(input):
                msgs_history = [{'role': 'user', 'content': [librosa.load(input, sr=16000, mono=True)[0]]}]
            else:
                msgs_history = [{'role': 'user', 'content': [input]}]
            
        self.model.reset_session()

        text_unpad, text_token, text_token_len, hidden_states = self.model.get_hidden(
            msgs=msgs_history,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
        )
        
        return text_unpad, text_token, text_token_len, hidden_states
        
sign_dict = {'。\n\n':3407, '。\n':8997, '。”':32945, '。“':53647, '”。':55807, '）。':74276, '。':1773, '！\n\n':17701, '！':6313, '？':11319,'[?]':30,'!':0, 
            '：\n':28311, '：':5122, ':\n':510, ':':25}
def postprocess(text_unpad, text_token, text_token_len, hidden_states, max_text_len=100):
    text_unpad_ori = copy.deepcopy(text_unpad)
    searched_ids = set()
    while len(text_unpad[0]) > max_text_len:
        for sign_id, sign in enumerate(sign_dict):
            search_res = re.search(sign, text_unpad[0])
            if search_res is None:
                continue
            search_ids = search_res.span()
            if search_ids[1]==len(text_unpad[0]):
                for i in range(search_ids[0],search_ids[1]):
                    searched_ids.add(i)
                continue
            if search_ids[0] in searched_ids:
                continue
            elif search_ids[1] in searched_ids:
                for i in range(search_ids[0],search_ids[1]):
                    searched_ids.add(i)
                continue
            else:
                for i in range(search_ids[0],search_ids[1]):
                    searched_ids.add(i)
            text_unpad = [text_unpad[0][:search_ids[0]]+'?'] if sign=='[?]' else [text_unpad[0][:search_ids[0]]+sign]
            try:
                token_id = text_token[0].cpu().numpy().tolist().index(sign_dict[sign])
                text_token = text_token[:, :token_id+1]
                # if not text_token[0].cpu().numpy().tolist() == chatsystem.tokenizer(text_unpad)['input_ids'][0]:
                if not chatsystem.tokenizer.decode(text_token[0].cpu().numpy().tolist()) == text_unpad[0]:
                    #判断clip后的text和token是否一致，不一致则返回状态码1
                    # import ipdb; ipdb.set_trace()
                    return 1, text_unpad_ori, text_unpad, text_token, text_token_len, None

            except:
                # import ipdb; ipdb.set_trace()
                #大概率是在text找到了sign, 但是token中没有对应的sign_dict[sign]，需要查看为什么，返回状态码2
                return 2, text_unpad_ori, text_unpad, text_token, text_token_len, None
            text_token_len[0] = token_id+1
            hidden_states = hidden_states[:, :token_id+1]
            break
        if sign_id == len(sign_dict)-1:
            break


    return 0, text_unpad_ori, text_unpad, text_token, text_token_len, hidden_states

def single_job(data, id):

    spk_id = "%06d" % id
    text_save_name = '{}/{}.json'.format(text_save_path, spk_id)
    text_token_save_name = '{}/{}.pt'.format(text_token_save_path, spk_id)
    hidden_states_save_name = '{}/{}.pt'.format(hidden_states_save_path, spk_id)
    if os.path.exists(text_save_name) and os.path.exists(text_token_save_name) and os.path.exists(hidden_states_save_name):
        return -1, None, None, None, None 

    question = data[id][0]["content"]
    text_unpad, text_token, text_token_len, hidden_states = chatsystem.OutputHidden(question)

    code, text_unpad_ori, text_unpad, text_token, text_token_len, hidden_states = postprocess(text_unpad, text_token, text_token_len, hidden_states)
    # print(text_unpad) #文本list=[str]
    
    print('~ '*10,f'id:{id} 大模型输出','~ '*10)
    print(f"question: {question}\ntext_unpad_ori: {text_unpad_ori[0]}\ntext_unpad: {text_unpad[0]}") 
    # print(text_unpad)#文本list=[str]
    # print(text_token.shape) #torch.Size([1, 76])
    # print(text_token_len) #tensor([76], dtype=torch.int32)
    # print(hidden_states.shape) #torch.Size([1, 76, 3584])
    print('~'*50)

    f = open(text_save_name, 'w')
    json.dump(text_unpad, f, ensure_ascii=False, indent=4)
    f.close()
    torch.save(text_token, text_token_save_name)
    torch.save(hidden_states, hidden_states_save_name)
    
    return id, text_unpad, text_token, text_token_len, hidden_states


def get_answer(data_file, save_path):

    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")

    all_task = [executor.submit(single_job, data, id) for id in range(data_num)]
    # all_task = [executor.submit(single_job, data, id) for id in range(10)]
    text_unpad_all = {}
    text_token_all = {}
    text_token_len_all = {}
    hidden_states_all = {}
    for future in tqdm(as_completed(all_task)):
        data_id, text_unpad, text_token, text_token_len, hidden_states = future.result()
        if data_id<0:
            continue
        spk_id = "%06d" % data_id
        text_unpad_all[spk_id] = text_unpad
        text_token_all[spk_id] = text_token
        text_token_len_all[spk_id] = text_token_len
        hidden_states_all[spk_id] = hidden_states

        
    # f = open('{}/text.json'.format(save_path), 'w')
    # json.dump(text_unpad_all, f, ensure_ascii=False, indent=4)
    # f.close()
    # torch.save(text_token_all, '{}/text_token.pt'.format(save_path))
    # torch.save(hidden_states_all, '{}/hidden_states.pt'.format(save_path))


chatsystem = AgentChat()
if __name__ == '__main__':
    data_root = "/mnt/afs/zhoufangru/agent/data/llm/RedGPT-main"
    data_file = os.path.join(data_root, "normalized_data.json")
    executor = ThreadPoolExecutor(max_workers=8)

    text_save_path = os.path.join(data_root, 'text')
    text_token_save_path = os.path.join(data_root, 'text_token')
    hidden_states_save_path = os.path.join(data_root, 'hidden_states')
    if not os.path.exists(text_save_path):
        os.makedirs(text_save_path)
    if not os.path.exists(text_token_save_path):
        os.makedirs(text_token_save_path)
    if not os.path.exists(hidden_states_save_path):
        os.makedirs(hidden_states_save_path)

    get_answer(data_file, data_root)

    