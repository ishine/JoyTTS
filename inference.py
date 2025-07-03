import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, 'third_party/Matcha-TTS'))
import torchaudio
import time
import torch
import random
import argparse
import numpy as np
import librosa
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
# from vllm import ModelRegistry
# from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
# ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

def infer(question, prompt_text, prompt_wav, stream=True, spk_id='', output_name=None, this_uuid=None):
    result = []
    for out in cosyvoice.inference_end2end(question, prompt_text, prompt_wav, stream=stream, speed=1.0, spk_id=spk_id, this_uuid=this_uuid):
        result.append(out['tts_speech'])
            
    text = out['text_answer'][0] if isinstance(out['text_answer'], list) else out['text_answer']
    if output_name is not None:
        sample_rate = cosyvoice.sample_rate
        result = torch.concat(result, dim=1)
        torchaudio.save(output_name, result, sample_rate)

        print(text)
        f = open(output_name.replace('.wav','.txt'), 'w')
        f.write(text)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, default='今天天气怎么样？')
    parser.add_argument('--prompt_wav', type=str, default='')
    parser.add_argument('--prompt_text', type=str, default='')
    parser.add_argument('--spk_id', type=str, default='孙悟空')
    parser.add_argument('--stream', type=bool, default=False)
    parser.add_argument('--output_name', type=str, default='output.wav')
    args = parser.parse_args()
    stream = args.stream
    if stream:
        cosyvoice = CosyVoice2(os.path.join(__dir__, 'pretrained_models'), load_jit=False, load_trt=False, load_vllm=False, fp16=False, use_flow_cache=True, end2end=True)
    else:
        cosyvoice = CosyVoice2(os.path.join(__dir__, 'pretrained_models'), load_jit=False, load_trt=False, load_vllm=False, fp16=False, use_flow_cache=False, end2end=True)

    infer(args.question, args.prompt_text, args.prompt_wav, stream=stream, spk_id=args.spk_id, output_name=args.output_name)
    
    # for spk_id in ["孙悟空", "猪八戒", "林黛玉"]:
    #     infer(args.question, args.prompt_text, args.prompt_wav, stream=stream, spk_id=spk_id, output_name=f"{spk_id}.wav")
