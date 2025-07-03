import os
import sys
from typing import Generator
import time
now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir, ".."))

import argparse
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from fastapi.responses import StreamingResponse
import logging
import torch
import urllib.request
# from config import TTS_Request
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import io
import base64

import warnings
warnings.filterwarnings("ignore")

__dir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import random
import json
import librosa
import torchaudio

cosyvoice = CosyVoice2(os.path.join(__dir__, 'pretrained_models'), load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    max_val = 0.8
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"Custom: {record.msg}"
        return super().format(record)
    
# 创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建一个处理程序并设置格式化器
logger.handlers = []
handler = logging.StreamHandler()
formatter = CustomFormatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

# 将处理程序添加到logger
logger.addHandler(handler)


app = FastAPI()
def numpy2base64(arr):
    bytesio = BytesIO()
    np.savetxt(bytesio, arr) # 只支持1维或者2维数组，numpy数组转化成字节流
    content = bytesio.getvalue()  # 获取string字符串表示
    b64_code = base64.b64encode(content).decode("utf-8")
    return b64_code


def single_job(tts_text, prompt_text, prompt_wav, utt, save_wav=False, save_token=False, save_root=None):
    
    prompt_speech_16k = postprocess(load_wav(prompt_wav, 16000))


    result_wav = []
    result_speech_token = []
    sample_rate = cosyvoice.sample_rate
    for out in cosyvoice.inference_extract_speech_token(tts_text, prompt_text, prompt_speech_16k, save_wav=save_wav):
        
        if save_wav:
            result_wav.append(out['tts_speech'])

        if save_token:
            speech_token = out['speech_token'] #torch.Size([1, 161])
            result_speech_token.append(speech_token)

    wav_base64, speech_token_base64 = None, None
    if save_wav:
        result_wav = torch.concat(result_wav, dim=1) #torch.Size([1, t])
        # torchaudio.save(f'{save_root}/{utt}.wav', result_wav, sample_rate)
        wav_base64 = numpy2base64(result_wav.numpy())

    if save_token:
        result_speech_token = torch.concat(result_speech_token, dim=1)#torch.Size([1, s])
        speech_token_base64 = numpy2base64(result_speech_token.numpy())

    return {'speech_token_base64':speech_token_base64, 'wav_base64':wav_base64, 'sample_rate':sample_rate}

    

async def tts_handle(req:dict):
    tts_text = req['tts_text']
    prompt_text = req['prompt_text']
    prompt_wav = req['prompt_wav'] #本地路径或者URL
    utt = req['utt']
    oss_day = req['oss_day'] if "oss_day" in req else None
    save_wav = req['save_wav'] if "save_wav" in req else False
    save_token = req['save_token'] if "save_token" in req else False

    # try:
    for i in range(1):
        content = single_job(tts_text, prompt_text, prompt_wav, utt, save_wav=save_wav, save_token=save_token)
        content["message"] = "success"
        result = JSONResponse(status_code=200, content=content)
        
    # except Exception as e:
    #     result = JSONResponse(status_code=400, content={"message": f"failed", "Exception": str(e)})
    
    return result
                
@app.post("/tts")
async def tts_post_endpoint(request: Request):
    params = await request.json()

    logging.info(f"tts_post_endpoint: {params}")
    return await tts_handle(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    # debug使用
    uvicorn.run(app="sever_tts:app", host="0.0.0.0", port=6667, workers=args.workers)
    # 部署使用
    # gunicorn sever_tts:app -w 4 -b 0.0.0.0:9880 --worker-class uvicorn.workers.UvicornWorker

