import os
import sys
from typing import Generator
import time
__dir__ = os.path.dirname(os.path.abspath(__file__))

import argparse
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
import logging
import torch
from cosyvoice.cli.cosyvoice import CosyVoice2
# from config import TTS_Request
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from test import infer

import warnings
warnings.filterwarnings("ignore")

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

chatsystem = CosyVoice2(os.path.join(__dir__, 'pretrained_models'), load_jit=False, load_trt=False, load_vllm=False, fp16=False, use_flow_cache=True, end2end=True)

def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


sample_rate = chatsystem.sample_rate

def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()

def streaming_generator(audio_generator:Generator, media_type:str):
    if media_type == "wav":
        yield wave_header_chunk()
        media_type = "raw"
    for out in audio_generator:
        chunk = out['tts_speech'].cpu().numpy()
        yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
        
async def llm_handle(req:dict):
    text = req["text"]
    role = req["role"]
    this_uuid = req["uuid"] if "uuid" in req else None
    audio = []
    tts_generator = chatsystem.inference_end2end(text, '', '', stream=True, zero_shot_spk_id=role, this_uuid=this_uuid)
        
    # return StreamingResponse(streaming_generator(tts_generator))
    return StreamingResponse(streaming_generator(tts_generator,'wav'), media_type="audio/wav")
                
@app.post("/llm")
async def tts_post_endpoint(request: Request):
    params = await request.json()

    logging.info(f"llm_post_endpoint: {params}")
    return await llm_handle(params)

if __name__ == "__main__":
    # debug使用
    uvicorn.run(app="model_sever:app", host="0.0.0.0", port=6666, workers=1)
    # 部署使用
    # gunicorn sever_llmanswer:app -w 1 -b 0.0.0.0:9880 --worker-class uvicorn.workers.UvicornWorker
