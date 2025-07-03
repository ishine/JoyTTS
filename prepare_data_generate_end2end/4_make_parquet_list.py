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
    save_utt_list, conversations_list, reference_list, speech_token_list, audio_data_list, wavinfo_list = [], [], [], [], [], []
    
    print(f"正在处理文件: {int(id_0/num_utts_per_parquet)+1}/{file_num}")
    file_path = os.path.join(data_root, llmanswer_file, data_name)
    file_path_ori = os.path.join(data_root, llmanswer_file, data_name.replace('_augment',''))
    save_tts_utt_list = []
    for id, data_info in tqdm(enumerate(data_info_list)):
        
        reference = data_info['data_info']["reference"] if "reference" in data_info['data_info'] else None
        conversations = data_info['data_info']['conversations']
        speech_token_conversation = [None]*int(len(conversations)/2)
        audio_data_conversation = [None]*int(len(conversations)/2)
        wavinfo_conversation = [None]*int(len(conversations)/2)
        data_utt = data_info['data_info']['utt']
        for i in range(0,len(conversations),2):
            con_id = '%02d'%i
            data_prefix = data_name.replace('_augment','') if data_name.endswith('_augment') else data_name
            llm_utt = f"{data_utt}_{con_id}" 
            # print(conversations, '{}/{}/{}.wav'.format(file_path, "wav", llm_utt))

            try:
                speech_token = torch.load('{}/{}/{}.pt'.format(file_path, "speech_token", llm_utt)) #(t)
                speech_token_conversation[int(i/2)] = speech_token.flatten().tolist()
                
                # audio_data = open('{}/{}/{}.wav'.format(file_path, "wav", llm_utt), 'rb').read()
                # audio_data_conversation[int(i/2)] = audio_data
                
                wavinfo = json.loads(open('{}/{}/{}.json'.format(file_path, "wavinfo", llm_utt), 'rb').read())
                wavinfo_conversation[int(i/2)] = wavinfo
                
            except:
                print('{}/{}/{}.pt 打开失败！！！！'.format(file_path, "speech_token", llm_utt))
                continue 
            
        save_utt_list.append(data_utt)
        conversations_list.append(conversations)
        reference_list.append(reference)
        speech_token_list.append(speech_token_conversation)
        audio_data_list.append(audio_data_conversation)
        wavinfo_list.append(wavinfo_conversation)
        
    # import ipdb; ipdb.set_trace()
    # 保存到parquet
    df = pd.DataFrame()
    df['utt'] = save_utt_list
    df['conversations'] = conversations_list
    df['reference'] = reference_list
    df['speech_token'] = speech_token_list #len(speech_token_list)=1000,[a1,...a1000],其中ai是当前对话n个回答的speech_token(type=tensor)
    # df['audio_data'] = audio_data_list
    df['wavinfo'] = wavinfo_list
    df.to_parquet(parquet_file)
    print(f'file saved in {parquet_file}, spend time {time.time() - start_time}')

def read_conversation(data_file):
    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")
    data_info_list = [{'id':i, 'data_info':data[i]} for i in range(data_num)]
    return data_info_list

'''
use_llmlabel:是否使用llm生成的结果作为tts的输入，True表示使用原始数据库的answer，False表示使用llm生成的结果
'''
use_llmlabel=True
llmanswer_file = "llmanswer_uselabel" if use_llmlabel else "llmanswer"
parquet_save_file = "parquet_uselabel" if use_llmlabel else "parquet"

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
    
    
    # mode_list = ['train']
    mode_list = ['train','eval']
    for mode in mode_list:
        pool = multiprocessing.Pool(processes=8)
        
        data_file = os.path.join(data_root, data_name, f"{mode}.json")
        
        des_dir = os.path.join(data_root, parquet_save_file, data_name, mode)
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)
    
        # retry_i_list = [int(line.strip()) for line in open('{}/data_error.list'.format(des_dir), 'r').readlines()]
        # retry_i_list = [132]
        retry_i_list = None
        
        
        data_info_list = read_conversation(data_file)
                
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
                
