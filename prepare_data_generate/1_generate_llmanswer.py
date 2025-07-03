# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/21 10:31
@Auth ： lixin749
@File ：test_offline.py
@Motto：ABC(Always Be Coding)

"""

# -*- coding:utf-8 -*-
import json
import requests
from time import time
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir, ".."))

ip_list = ['0.0.0.0']
port = '6666'

local_name_list = [f"{ip}:{port}" for ip in ip_list]

def save_result(conversation, result, spk_id, save_oss=False):
    
    save_text_name = os.path.join(save_text_path, spk_id+".json")
    save_hidden_name = os.path.join(save_hidden_path, spk_id+".pt")
    
    result["question"] = conversation[-1]["content"]

    hidden_states = result.pop('hidden_states')
    hidden_states = torch.tensor(hidden_states).view(-1,3584)
    print(result["question"], result['llm_answer'], result['tts_input'], np.array(result['text_token']).shape, hidden_states.shape)
    
    # import ipdb; ipdb.set_trace()
    f = open(save_text_name, 'w')
    json.dump(result, f, ensure_ascii=False, indent=4)
    f.close()

    torch.save(hidden_states, save_hidden_name)

    if save_oss:
        url_in_text, url_out_text = upload_file(os.path.abspath(save_text_name), prefix="agent/data/llmanswer/text", expires_in_days=30)
        url_in_hidden, url_out_hidden = upload_file(os.path.abspath(save_hidden_name), prefix="agent/data/llmanswer/hidden_states", expires_in_days=30)

def chatsystem(conversation, local_name, spk_id):
    t1 = time()
    body = {"conversation": conversation}
    headers = {'Content-Type': 'application/json'}
    datas = json.dumps(body)
    response = requests.post(f"http://{local_name}/llm", data=datas, headers=headers)
    t2 = time()
    all_time = t2 - t1
    # print("耗时：{:.3f}".format(all_time))
    result = response.json()
    if result["message"] == "success":
        save_result(conversation, result, spk_id)
        
    else:
        print(result)
        

def single_job(data, data_name, id):
    local_name_len = len(local_name_list)
    local_name = local_name_list[id%local_name_len]

    reference = data[id]["reference"] if "reference" in data[id] else None
    conversations = data[id]["conversations"]
    # assert len(conversations)%2 == 0, f"conversations error: {conversations}"

    for i in range(0,len(conversations),2):
        data_id='%06d'%id
        con_id = '%02d'%i
        spk_id = f"{data_name}_{data_id}_{con_id}"
        # save_json_name = spk_id+".json"
        # if exists_file(f"agent/data/llmanswer/hidden_states/{spk_id}.pt"):
        if os.path.exists(os.path.join(save_hidden_path, spk_id+".pt")):
            # print(f"spk_id: {spk_id} 已存在")
            continue
        conversation = conversations[:i+1] if reference is None else [{"role": "system","content":reference}] + conversations[:i+1]
        chatsystem(conversation, local_name, spk_id)



def get_answer(data_file, data_name):

    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")

    all_task = [executor.submit(single_job, data, data_name, id) for id in range(data_num)]
    # all_task = [executor.submit(single_job, data, data_name, id) for id in range(1)]
    text_unpad_all = {}
    text_token_all = {}
    text_token_len_all = {}
    hidden_states_all = {}
    for future in tqdm(as_completed(all_task)):
        pass
    
def save_result_fromoss(data_file, data_name, save_path, downfile_list=["text","hidden_states"]):

    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")

    for downfile in downfile_list:
        save_file_path = os.path.join(save_path, downfile)
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)

        for id in tqdm(range(data_num)):
        # for id in range(10):
            # reference = data[id]["reference"]
            conversations = data[id]["conversations"]
            # assert len(conversations)%2 == 0, f"conversations error: {conversations}"
            url_out_list = []
            for i in range(0,len(conversations),2):
                data_id='%06d'%id
                con_id = '%02d'%i
                spk_id = f"{data_name}_{data_id}_{con_id}"
                save_file_name = spk_id+".json" if downfile=="text" else spk_id+".pt"
                save_name = os.path.join(save_file_path, save_file_name)
                if os.path.exists(save_name):
                    # print(f"save_name: {save_name} 已存在")
                    continue
                
                oss_key_name = os.path.join("agent/data/llmanswer", downfile, save_file_name)
                download_file(oss_key_name, save_file_path, net='in')
                
def cal_data_num(data_file, data_name):

    data = json.load(open(data_file, 'r'))
    data_num = len(data)

    count = 0
    for id in range(data_num):
        conversations = data[id]["conversations"]
        for i in range(0,len(conversations),2):
            count += 1
    print(f"get conversation number {data_num}, text number {count}")
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../../data/llm/")
    parser.add_argument('--data_name', type=str, default="RedGPT-main")
    args = parser.parse_args()

    executor = ThreadPoolExecutor(max_workers=8)
    data_root = args.data_root
    data_name = args.data_name
    save_text_path = os.path.join(data_root, "llmanswer", data_name, "text")
    save_hidden_path = os.path.join(data_root, "llmanswer", data_name, "hidden_states")
    if not os.path.exists(save_text_path):
        os.makedirs(save_text_path)
    if not os.path.exists(save_hidden_path):
        os.makedirs(save_hidden_path)
    
    data_file = os.path.join(data_root, data_name, "normalized_data.json")

    get_answer(data_file, data_name)

    ##单任务debug
    # data = json.load(open(data_file, 'r'))
    # data_num = len(data)
    # print(f"get data number {data_num}")
    # single_job(data, data_name, 0)

    ##将oss上的json下载解析，并保存为本地文件
    # save_path = os.path.join(data_root, "llmanswer")
    # save_result_fromoss(data_file, data_name, save_path, downfile_list=["text","hidden_states", "speech_token"])
    # save_result_fromoss(data_file, data_name, save_path, downfile_list=["speech_token"])

    ##计算文本数量
    # cal_data_num(data_file, data_name)
    