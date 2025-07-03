import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch


def read_conversation(data_file):
    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")
    data_info_list = [{'id':i, 'data_info':data[i]} for i in range(data_num)]
    return data_info_list


if __name__ == "__main__":
    llmanswer_file = "llmanswer_uselabel"
    data_root = "/media/cfs/zhoufangru/workspace/agent/data/llm"
    
    data_name = "RedGPT-main"
    mode = 'train'
    data_file = os.path.join(data_root, data_name, f"{mode}.json")

    data_info_list = read_conversation(data_file)
    time = 0
    for id, data_info in tqdm(enumerate(data_info_list)):
        
        conversations = data_info['data_info']['conversations']
        data_utt = data_info['data_info']['utt']
        for i in range(0,len(conversations),2):
            con_id = '%02d'%i
            llm_utt = f"{data_utt}_{con_id}" 
            
            file_name = os.path.join(data_root, llmanswer_file, data_name, "wavinfo", f"{llm_utt}.json")
            if not os.path.exists(file_name):
                continue
            wavinfo = json.loads(open(file_name, 'rb').read())
            time += wavinfo['wav_duration']
    print(time/3600)


