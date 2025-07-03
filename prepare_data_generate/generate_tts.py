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
from concurrent.futures import ThreadPoolExecutor, as_completed

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir, ".."))
from oss import upload_file, download_file, exists_file

# ip = '6.61.22.41'
# ip = '0.0.0.0'
ip_list = ['6.19.152.219'] + [f'6.61.22.{i}' for i in range(62,84)] + [f'6.61.22.{i}' for i in range(218,234)]
# ip_list = ['0.0.0.0']
port = '6666'
# port_list = ['6666']

# local_name = f"{ip}:{port}"
local_name_list = [f"{ip}:{port}" for ip in ip_list]
# local_name_list = [f"{ip}:{port}" for port in port_list]

def tts(tts_text, prompt_text, prompt_wav, spk_id, local_name):
    t1 = time()
    body = {"tts_text": tts_text, "prompt_text": prompt_text, "prompt_wav": prompt_wav, "utt":spk_id,"save_wav":save_wav}
    headers = {'Content-Type': 'application/json'}
    datas = json.dumps(body)
    response = requests.post(f"http://{local_name}/tts", data=datas, headers=headers)
    t2 = time()
    all_time = t2 - t1
    # print("耗时：{:.3f}".format(all_time))
    result = response.json()

    if result["message"] == "success":
        b64_code = result["b64_code"]
        sample_rate = result["sample_rate"]
        save_result(spk_id, b64_code, sample_rate, save_wav=save_wav)
    else:
        print(result)
        return ""
    
def save_result(spk_id, b64_code, sample_rate, save_wav=False, oss_day=30, save_oss=False):
    b64_decode = base64.b64decode(b64_code.encode('utf-8'))
    result_numpy = np.loadtxt(BytesIO(b64_decode))
    result_torch = torch.from_numpy(result_numpy).to(torch.int64)
    if save_wav:
        save_wav_name = os.path.join(save_wav_path, f"{spk_id}.wav")
        result_torch = result_torch.view(1,-1)
        torchaudio.save(save_wav_name, result_torch, sample_rate)

    else:
        save_speech_name = os.path.join(save_speech_path, f"{spk_id}.pt")
        torch.save(result_torch, save_speech_name)

    
    if save_oss:
        url_in, url_out = upload_file(os.path.abspath(save_wav_name), prefix="agent/data/llmanswer/wav", expires_in_days=oss_day)
        url_in, url_out = upload_file(os.path.abspath(save_speech_name), prefix="agent/data/llmanswer/speech_token", expires_in_days=oss_day)
        


save_wav = True

save_wav_path = "result_tts"
if not os.path.exists(save_wav_path):
    os.makedirs(save_wav_path)


if __name__ == '__main__':
    
    prompt_wav = 'https://medical-consultation.s3-internal.cn-north-1.jdcloud-oss.com/agent/data/audio/huangyufeng_ref.WAV'
    prompt_text = '其实众所周知啊，不管是我。'
    tts_text = ''.join([line for line in open('text.txt','r').readlines()])
    spk_id = 26
    local_name = local_name_list[0]
    tts(tts_text, prompt_text, prompt_wav, spk_id, local_name)
