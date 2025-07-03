import json
from time import time
import os
import sys
from tqdm import tqdm
import numpy as np
import random
import argparse


def gen_data():
    data_file = os.path.join(data_root, data_name, "normalized_data.json")

    save_list_path = os.path.join(data_root, "llmanswer", data_name, "data.list")
    f = open(save_list_path, 'w')
    
    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")
    text_num = 0
    for id in tqdm(range(data_num)):
        conversations = data[id]["conversations"]
        # assert len(conversations)%2 == 0, f"conversations error: {conversations}"
    
        for i in range(0,len(conversations),2):
            data_id='%06d'%id
            con_id = '%02d'%i
            spk_id = f"{data_name.replace('_augment','')}_{data_id}_{con_id}" if data_name.endswith("_augment") else f"{data_name}_{data_id}_{con_id}"

            tts_spk_id_list = gen_pt_data(spk_id) if data_name.endswith("_augment") else [spk_id]

            for tts_spk_id in tts_spk_id_list:
                f.write(tts_spk_id+'\n')
                text_num += 1

    f.close()
    print(f"get text number {text_num}")
    
def gen_pt_data(spk_id):
    llm_json_name = spk_id+".json"
    text_json_path = os.path.join(data_root, "llmanswer", data_name, "text", llm_json_name)
    if not os.path.exists(text_json_path):
        print(f"text_json_path: {text_json_path} 不存在")
        return []
            
    ##读取本地的llm生成的tts_answer
    tts_text_list = json.load(open(text_json_path, 'r'))['tts_input']
    # print("tts_text_list: ", tts_text_list)
    text_num = len(tts_text_list)
    tts_spk_id_list = []
    for j, tts_text in enumerate(tts_text_list):

        text_id = '%02d'%j
        tts_spk_id = f"{spk_id}_{text_id}"

        tts_spk_id_list.append(tts_spk_id)
    return tts_spk_id_list
            
                
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../../data/llm/")
    parser.add_argument('--data_name', type=str, default="RedGPT-main")
    args = parser.parse_args()

    data_root = args.data_root
    data_name = args.data_name

    gen_data()