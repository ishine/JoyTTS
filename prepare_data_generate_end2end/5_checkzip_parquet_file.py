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

def check_df(parquet_file, id):
    data_count = 0
    if not os.path.exists(parquet_file):
        print(f"parquet_file:{parquet_file}不存在")
        return False, id
    try:
        for df in pq.ParquetFile(parquet_file).iter_batches(batch_size=64):
            df = df.to_pandas()
            data_count += len(df)
        print(f"parquet_file:{parquet_file}打开成功，数据量{data_count}")
        return True, id
    except:
        print(f"parquet_file:{parquet_file}打开失败")
        return False, id
        
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
    
    data_name = "RedGPT-main"
    # data_name = "RedGPT-main_augment"
    # data_name = "generated_chat_0.4M"
    # data_name = "generated_chat_0.4M_augment"
    num_utts_per_parquet=1000 if data_name.endswith('_augment') else 250

    data_root = "/media/cfs/zhoufangru/workspace/agent/data/llm"
    # mode_list = ['train','eval']
    mode_list = ['train']
    
    for mode in mode_list:
        pool = multiprocessing.Pool(processes=10)
        des_dir = os.path.join(data_root, parquet_save_file, data_name, mode)
        
        f_error = open('{}/data_error.list'.format(des_dir), 'w')


        data_file = os.path.join(data_root, data_name, f"{mode}.json")
        data_info_list = read_conversation(data_file)
                
        file_num = int(len(data_info_list)/num_utts_per_parquet)+1
        
        results = []
        file_start = 0
        file_end = file_num
        for i in range(file_start, file_end):
            parquet_file = os.path.join(des_dir, 'parquet_{:09d}.tar'.format(i))
            results.append(pool.apply_async(check_df, (parquet_file, i)))
            
            # suc = check_df(parquet_file)

        pool.close()
        pool.join()

        for rec in results:
            suc, id = rec.get()
            if not suc:
                f_error.write(str(id)+'\n')
                
            
        f_error.close()
