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
# from config import TTS_Request
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from prepare_data_sever.prepare_llmanswer_offline import chatsystem, postprocess

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



async def llm_handle(req:dict):
    try:
    # for i in range(1):
        text_unpad, text_token, text_token_len, hidden_states = chatsystem.OutputHidden(req["conversation"])
        max_text_len = req["max_text_len"] if 'max_text_len' in req else 100
        suc, text_unpad_ori, text_unpad, text_token, text_token_len, hidden_states = postprocess(text_unpad, text_token, text_token_len, hidden_states, max_text_len=max_text_len)
        if suc==0:
            return JSONResponse(status_code=200, content={"message": "success", "llm_answer": text_unpad_ori, "tts_input": text_unpad, \
                "text_token":text_token.cpu().numpy().flatten().tolist(), "text_token_len":text_token_len.numpy().flatten().tolist(), \
                    "hidden_states":hidden_states.cpu().to(dtype=torch.float32).numpy().flatten().tolist()})
        else:
            return JSONResponse(status_code=200, content={"message": "failed",  "Exception": f"code:{suc}", "llm_answer": text_unpad_ori, "tts_input": text_unpad, \
                "text_token":text_token.cpu().numpy().flatten().tolist(), "text_token_len":text_token_len.numpy().flatten().tolist(), \
                    "hidden_states":None})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"failed", "Exception": str(e)})
                
@app.post("/llm")
async def tts_post_endpoint(request: Request):
    params = await request.json()

    logging.info(f"llm_post_endpoint: {params}")
    return await llm_handle(params)

if __name__ == "__main__":
    # debug使用
    uvicorn.run(app="sever_llmanswer:app", host="0.0.0.0", port=6666, workers=1)
    # 部署使用
    # gunicorn sever_llmanswer:app -w 1 -b 0.0.0.0:9880 --worker-class uvicorn.workers.UvicornWorker
