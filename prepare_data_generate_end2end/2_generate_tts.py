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
# from oss import upload_file, download_file, exists_file, local_path_in

ip_list = ['0.0.0.0']
port_list = ['6667']

# local_name = f"{ip}:{port}"
local_name_list = [f"{ip}:{port}" for ip in ip_list for port in port_list]
# local_name_list = [f"{ip}:{port}" for port in port_list]
local_name_len = len(local_name_list)

def tts(tts_text, spk_id, prompt_text=None, prompt_wav=None):
    local_name = random.sample(local_name_list,1)[0]

    if prompt_text is None or prompt_wav is None:
        prompt_text, prompt_wav = get_prompt(debug=True)

    t1 = time()
    body = {"tts_text": tts_text, "prompt_text": prompt_text, "prompt_wav": prompt_wav, "utt":spk_id,"save_wav":save_wav, "save_token":save_token}
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
    
def save_result(spk_id, result, save_wav=True, save_token=False, oss_day=30, save_oss=False):
    
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

        # prompt_wav_oss = os.path.join('agent/data', prompt_wav[prompt_wav.index("WenetSpeech4TTS/Basic/"):])
        # prompt_wav = f'https://{local_path_in}/{prompt_wav_oss}'
        
    return prompt_text, prompt_wav


def get_token(data_file, data_name):
    '''
    use_llmlabel:是否使用llm生成的结果作为tts的输入，True表示使用原始数据库的answer，False表示使用llm生成的结果
    '''
    all_task = []
    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")
    print("正在收集任务...")
    for id in tqdm(range(data_num)):
    # for id in tqdm(range(5)):
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
    
            text_json_path = os.path.join(data_root, gen_save_file, data_name, "text", llm_json_name)
            if use_llmlabel:
                if i == len(conversations)-1:
                    # 当前Q没有对应的A
                    continue
                tts_text_list = [conversations[i+1]['content']]
            else:
                if not os.path.exists(text_json_path):
                    print(f"text_json_path: {text_json_path} 不存在")
                    continue
                ##读取本地的llm生成的tts_answer
                tts_text_list = json.load(open(text_json_path, 'r'))['tts_input']
    
    
            # print("tts_text_list: ", tts_text_list)
            text_num = len(tts_text_list)
            for j, tts_text in enumerate(tts_text_list):
                
                text_id = '%02d'%j
                tts_spk_id = f"{data_prefix}_{data_id}_{con_id}_{text_id}" if data_name.endswith('_augment') else llm_spk_id
        
                if save_wav:
                    save_name = os.path.join(save_wav_path, f"{tts_spk_id}.wav")
                else:
                    save_name = os.path.join(save_speech_path, f"{tts_spk_id}.pt")
                if os.path.exists(save_name):
                    # print(f"tts_spk_id: {tts_spk_id} 已存在")
                    continue
    
                # print(local_name)
    
                # all_task.append(executor.submit(tts, tts_text, tts_spk_id))
                tts(tts_text, tts_spk_id)

            
    # print(f"收集{len(all_task)}数据需要处理，开始生成...")
    # for future in tqdm(as_completed(all_task)):
    #     pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../../data/llm/")
    parser.add_argument('--data_name', type=str, default="RedGPT-main")
    parser.add_argument('--save_wav', type=bool, default=True)
    parser.add_argument('--save_token', type=bool, default=True)
    parser.add_argument('--prompt_root', type=str, default="../../data/tts_clean/00/WenetSpeech4TTS/Basic/")
    args = parser.parse_args()
    
    data_root = args.data_root
    data_name = args.data_name
    save_wav = args.save_wav
    save_token = args.save_token
    prompt_root = args.prompt_root
    # executor = ThreadPoolExecutor(max_workers=16)
    data_file = os.path.join(data_root, data_name, "normalized_data.json")
    
    '''
    use_llmlabel:是否使用llm生成的结果作为tts的输入，True表示使用原始数据库的answer，False表示使用llm生成的结果
    '''
    use_llmlabel=True

    gen_save_file = "llmanswer_uselabel" if use_llmlabel else "llmanswer"


    save_speech_path = os.path.join(data_root, gen_save_file, data_name, "speech_token")
    if not os.path.exists(save_speech_path):
        os.makedirs(save_speech_path)
    save_wav_path = os.path.join(data_root, gen_save_file, data_name, "wav")
    if not os.path.exists(save_wav_path):
        os.makedirs(save_wav_path)
    save_wavinfo_path = os.path.join(data_root, gen_save_file, data_name, "wavinfo")
    if not os.path.exists(save_wavinfo_path):
        os.makedirs(save_wavinfo_path)
    utt2wav, utt2text, utts, utts_num = load_prompt_info()


    get_token(data_file, data_name)



