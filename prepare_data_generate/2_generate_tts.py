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
import random
from io import BytesIO
import base64
import torch
import torchaudio
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir, ".."))

ip_list = ['0.0.0.0']
port = '6667'

local_name_list = [f"{ip}:{port}" for ip in ip_list]

def tts(tts_text, prompt_text, prompt_wav, spk_id, local_name):
    t1 = time()
    body = {"tts_text": tts_text, "prompt_text": prompt_text, "prompt_wav": prompt_wav, "utt":spk_id,"save_wav":save_wav,"save_token":save_token}
    headers = {'Content-Type': 'application/json'}
    datas = json.dumps(body)
    response = requests.post(f"http://{local_name}/tts", data=datas, headers=headers)
    t2 = time()
    all_time = t2 - t1
    # print("耗时：{:.3f}".format(all_time))
    result = response.json()
    if result["message"] == "success":
        result['tts_text'] = tts_text
        save_result(spk_id, result, save_wav=save_wav, save_token=save_token)
    else:
        print(spk_id, tts_text, result)
        return ""
    
def save_result(spk_id, result, save_wav=False, save_token=True, oss_day=30, save_oss=False):
    
    if save_wav:
        sample_rate = result['sample_rate']
        b64_decode = base64.b64decode(result['wav_base64'].encode('utf-8'))
        result_numpy = np.loadtxt(BytesIO(b64_decode))
            
        save_wav_name = os.path.join(save_wav_path, f"{spk_id}.wav")
        result_torch = torch.from_numpy(result_numpy).to(torch.float32).view(1,-1)
        torchaudio.save(save_wav_name, result_torch, sample_rate)
        
        save_wavinfo_name = os.path.join(save_wavinfo_path, f"{spk_id}.json")
        wavinfo = {'wav_frame':result_torch.shape[1], 'sample_rate':sample_rate, 'wav_duration':result_torch.shape[1]/sample_rate, 'text':result['tts_text']}
        f = open(save_wavinfo_name, 'w')
        json.dump(wavinfo, f, ensure_ascii=False, indent=4)
        f.close()

    if save_token:
        b64_decode = base64.b64decode(result['speech_token_base64'].encode('utf-8'))
        result_numpy = np.loadtxt(BytesIO(b64_decode))
        result_torch = torch.from_numpy(result_numpy).to(torch.int64)
        
        save_speech_name = os.path.join(save_speech_path, f"{spk_id}.pt")
        torch.save(result_torch, save_speech_name)

    
    if save_oss:
        url_in, url_out = upload_file(os.path.abspath(save_wav_name), prefix=f"agent/data/{gen_save_file}/wav", expires_in_days=oss_day)
        url_in, url_out = upload_file(os.path.abspath(save_speech_name), prefix=f"agent/data/{gen_save_file}/speech_token", expires_in_days=oss_day)
        


def load_prompt_info():
    prompt_text_file = os.path.join(prompt_root, "text")
    prompt_wav_file = os.path.join(prompt_root, "wav.scp")
    utt2wav = {}
    with open(prompt_wav_file) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    utt2text = {}
    with open(prompt_text_file) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2text[l[0]] = ' '.join(l[1:])

    utts = list(utt2wav.keys())
    utts_num = len(utts)
    return utt2wav, utt2text, utts, utts_num


def get_prompt(debug=False):
    if debug:
        prompt_text = '齐天大圣下凡，与两位师弟共保大唐高僧，西天取经的。'
        prompt_wav = '../../data/audio/孙悟空.WAV'
    else:
        ##随机选一个prompt
        random_id = random.randint(0,utts_num)
        prompt_utt = utts[random_id]
        prompt_text = utt2text[prompt_utt]
        prompt_wav = utt2wav[prompt_utt]
        # prompt_wav = os.path.join(data_root.replace("llm","tts_clean"),prompt_wav[prompt_wav.index("WenetSpeech4TTS/Basic/"):])
        prompt_wav = os.path.join(data_root.replace("llm","tts_clean"), '00',prompt_wav[prompt_wav.index("WenetSpeech4TTS/Basic/"):])

    return prompt_text, prompt_wav

def single_job(data, data_name, id):
    local_name_len = len(local_name_list)
    local_name = local_name_list[id%local_name_len]

    # reference = data[id]["reference"]
    conversations = data[id]["conversations"]
    # assert len(conversations)%2 == 0, f"conversations error: {conversations}"
    url_out_list = []
    for i in range(0,len(conversations),2):
        data_id='%06d'%id
        con_id = '%02d'%i
        data_prefix = data_name.replace('_augment','') if data_name.endswith('_augment') else data_name
        llm_spk_id = f"{data_prefix}_{data_id}_{con_id}" 
        llm_json_name = llm_spk_id+".json"
            
        text_json_path = os.path.join(data_root, "llmanswer", data_name, "text", llm_json_name)
        if not os.path.exists(text_json_path):
            print(f"text_json_path: {text_json_path} 不存在")
            continue
            
        ##读取本地的llm生成的tts_answer
        tts_text_list = json.load(open(text_json_path, 'r'))['tts_input']
        # print("tts_text_list: ", tts_text_list)
        text_num = len(tts_text_list)
        for j, tts_text in enumerate(tts_text_list):
            if tts_text.replace(' ', '') == '《':
                continue
            
            text_id = '%02d'%j
            tts_spk_id = f"{data_prefix}_{data_id}_{con_id}_{text_id}" if data_name.endswith('_augment') else llm_spk_id
    
            if save_wav:
                save_name = os.path.join(save_wav_path, f"{tts_spk_id}.wav")
            else:
                save_name = os.path.join(save_speech_path, f"{tts_spk_id}.pt")
            if os.path.exists(save_name):
                print(f"tts_spk_id: {tts_spk_id} 已存在")
                continue
                
            time1 = time()
            ##随机选一个prompt
            prompt_text, prompt_wav = get_prompt(debug=False)

            tts(tts_text, prompt_text, prompt_wav, tts_spk_id, local_name)

        



def get_token(data_file, data_name):

    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")

    # all_task = [executor.submit(single_job, data, data_name, id) for id in range(data_num)]
    all_task = [executor.submit(single_job, data, data_name, id) for id in range(10)]
    text_unpad_all = {}
    text_token_all = {}
    text_token_len_all = {}
    hidden_states_all = {}
    for future in tqdm(as_completed(all_task)):
        pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../../data/llm/")
    parser.add_argument('--data_name', type=str, default="RedGPT-main")
    parser.add_argument('--save_wav', type=bool, default=False)
    parser.add_argument('--save_token', type=bool, default=True)
    parser.add_argument('--prompt_root', type=str, default="../../data/tts_clean/00/WenetSpeech4TTS/Basic/")
    args = parser.parse_args()
    
    data_root = args.data_root
    data_name = args.data_name
    save_wav = args.save_wav
    save_token = args.save_token
    prompt_root = args.prompt_root
    executor = ThreadPoolExecutor(max_workers=32)
    data_file = os.path.join(data_root, data_name, "normalized_data.json")

    save_speech_path = os.path.join(data_root, "llmanswer", data_name, "speech_token")
    if not os.path.exists(save_speech_path):
        os.makedirs(save_speech_path)
    save_wav_path = os.path.join(data_root, "llmanswer", data_name, "wav")
    if not os.path.exists(save_wav_path):
        os.makedirs(save_wav_path)
    save_wavinfo_path = os.path.join(data_root, "llmanswer", data_name, "wavinfo")
    if not os.path.exists(save_wavinfo_path):
        os.makedirs(save_wavinfo_path)
    utt2wav, utt2text, utts, utts_num = load_prompt_info()

    get_token(data_file, data_name)

    # data = json.load(open(data_file, 'r'))
    # data_num = len(data)
    # print(f"get data number {data_num}")
    # single_job(data, data_name, 0)


