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
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch
import pyarrow.parquet as pq

def check_df(parquet_file):
    data_count = 0
    try:
        for df in pq.ParquetFile(parquet_file).iter_batches(batch_size=64):
            df = df.to_pandas()
            data_count += len(df)
        print(f"parquet_file:{parquet_file}打开成功，数据量{data_count}")
    except:
        print(f"parquet_file:{parquet_file}打开失败")
    return data_count
        
def collect_data_set(tts_utt_list):
    llm_tts_map = {}
    if not data_name.endswith('_augment'):
        for tts_utt in tts_utt_list:
            llm_tts_map[tts_utt] = [tts_utt]
    else:
        for tts_utt in tts_utt_list:
            llm_utt = '_'.join(tts_utt.split('_')[:-1])
            if llm_utt in llm_tts_map:
                llm_tts_map[llm_utt].append(tts_utt)
            else:
                llm_tts_map[llm_utt] = [tts_utt]
    return llm_tts_map
            
        
    
def job(data_info_list, parquet_file, file_num):
    start_time = time.time()
    id_0 = data_info_list[0]["id"]
    text_list, text_token_list, speech_token_list, hidden_states_list = [], [], [], []
    tts_utt_list = [data_info['utt'] for data_info in data_info_list]
    llm_tts_map = collect_data_set(tts_utt_list)
    print(f"正在处理文件: {int(id_0/num_utts_per_parquet)+1}/{file_num}, 数据量:[llm:{len(llm_tts_map)}]-[tts:{len(tts_utt_list)}]")
    file_path = os.path.join(data_root, "llmanswer", data_name)
    file_path_ori = os.path.join(data_root, "llmanswer", data_name.replace('_augment',''))
    save_tts_utt_list = []
    for id, llm_utt in tqdm(enumerate(llm_tts_map.keys())):
        
        text_info = json.load(open('{}/{}/{}.json'.format(file_path, "text", llm_utt), 'r'))
        hidden_states_llm = torch.load('{}/{}/{}.pt'.format(file_path_ori, "hidden_states", llm_utt)) #(s'_llm, 3584)
                
        for tts_utt in llm_tts_map[llm_utt]:
            
            tts_id = int(tts_utt.split('_')[-1]) if data_name.endswith('_augment') else 0
            
            if isinstance(text_info, list):
                text = text_info[tts_id]  #(s)
                text_token = torch.load('{}/{}/{}.pt'.format(file_path, "text_token", llm_utt), weights_only=False)[tts_id].cpu().flatten().tolist() #(s')
            elif isinstance(text_info, dict):
                text = text_info["tts_input"][tts_id]  #(s)
                text_token = text_info["text_token"][tts_id] if isinstance(text_info["text_token"][0], list) else text_info["text_token"] #(s')
            
            clip_id = text_info['id_list'][tts_id] if data_name.endswith('_augment') else [0, len(text_token)]
            # import ipdb; ipdb.set_trace()
            # speech_token = torch.load('{}/{}/{}.pt'.format(file_path, "speech_token", tts_utt), weights_only=False)[0] #(t)
            # hidden_states = torch.load('{}/{}/{}.pt'.format(file_path, "hidden_states", llm_utt), weights_only=False)[0] #(s', 3584)
            try:
                speech_token = torch.load('{}/{}/{}.pt'.format(file_path, "speech_token", tts_utt)) #(t)
            except:
                print('{}/{}/{}.pt 打开失败！！！！'.format(file_path, "speech_token", tts_utt))
                continue 
            hidden_states = hidden_states_llm[clip_id[0]:clip_id[1]] #(s', 3584)
            assert len(text_token) == len(hidden_states), "feature align error"
            save_tts_utt_list.append(tts_utt)
            text_list.append(text)
            text_token_list.append(text_token)
            speech_token_list.append(speech_token.flatten().tolist())
            hidden_states_list.append(hidden_states.flatten().tolist())
    
            # if len(text_token)/len(speech_token)<0.03 or len(text_token)/len(speech_token)>0.3:
            #     print(f"text_token:{len(text_token)}, speech_token:{len(speech_token)}, ratio: {len(text_token)/len(speech_token)}, text: {text}")

    # 保存到parquet
    df = pd.DataFrame()
    df['utt'] = save_tts_utt_list
    df['text'] = text_list
    df['speech_token'] = speech_token_list #len(speech_token_list)=1000,[a1,...a1000],其中ai是speech_token(type=tensor)
    df['text_token'] = text_token_list #len(text_token_list)=1000,[a1,...a1000],其中ai是text_token(type=tensor)
    df['hidden_states'] = hidden_states_list #len(hidden_states_list)=1000,[a1,...a1000],其中ai是hidden_states(type=tensor)
    df.to_parquet(parquet_file)
    print(f'file saved in {parquet_file}, spend time {time.time() - start_time}')


if __name__ == "__main__":
    # Using process pool to speedup
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../../data/llm")
    parser.add_argument('--data_name', type=str, default="RedGPT-main")
    parser.add_argument('--num_utts_per_parquet', type=int, default=1000)
    args = parser.parse_args()
    
    data_root = args.data_root
    data_name = args.data_name
    num_utts_per_parquet = args.num_utts_per_parquet

    num_utts_per_parquet=4000 if data_name.endswith('_augment') else 1000
    
    mode_list = ['train', 'eval']
    for mode in mode_list:
        pool = multiprocessing.Pool(processes=16)
        
        src_dir = os.path.join(data_root, "llmanswer", data_name, f"{mode}.txt")
        des_dir = os.path.join(data_root, "parquet", data_name, mode)
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)
    
        # retry_i_list = [int(line.strip()) for line in open('{}/data_error.list'.format(des_dir), 'r').readlines()]
        # retry_i_list = [132]
        retry_i_list = None
        
        
        data_info_list = []
        with open(src_dir) as f:
            for l_id, l in enumerate(f):
                data_name_, utt = l.strip().split('\t')
                data_info_list.append({'utt':utt, 'id':l_id,'data_name':data_name})     
                
        file_num = int(len(data_info_list)/num_utts_per_parquet)+1
        print(f"数据集{mode}数据量{len(data_info_list)},被分成{file_num}个文件")
        if retry_i_list:
            print('注意正在重新生成失败文件...')
        parquet_list = []
        for i, j in enumerate(range(0, len(data_info_list), num_utts_per_parquet)):
            parquet_file = os.path.join(des_dir, 'parquet_{:09d}.tar'.format(i))
            if retry_i_list and i not in retry_i_list:
                continue
            elif retry_i_list is None and os.path.exists(parquet_file): # and check_df(parquet_file) == len(data_info_list):
                print(f"文件{parquet_file}已存在")
                continue
            parquet_list.append(parquet_file)
            pool.apply_async(job, (data_info_list[j: j + num_utts_per_parquet], parquet_file, file_num))
            # job(data_info_list[j: j + num_utts_per_parquet], parquet_file, file_num)
        pool.close()
        pool.join()
    
        if retry_i_list is None:
            with open('{}/data.list'.format(des_dir), 'w', encoding='utf8') as f1:
                for name in parquet_list:
                    f1.write(name.replace(data_root, 'data') + '\n')
                
