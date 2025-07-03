import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import time
from io import BytesIO
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.utils.common import set_all_random_seed

from cosyvoice.cli.cosyvoice import CosyVoice2
chatsystem = CosyVoice2(os.path.join(ROOT_DIR, 'pretrained_models'), load_jit=False, load_trt=False, load_vllm=False, fp16=False, use_flow_cache=True, end2end=True)
sample_rate = chatsystem.sample_rate

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def normalize_audio(audio):
    """
    Normalize audio array to be between -1 and 1
    :param audio: Input audio array
    :return: Normalized audio array
    """
    audio = np.clip(audio, -1, 1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def pack_audio(io_buffer:BytesIO, data:np.ndarray):

    io_buffer.write(data.tobytes())
    io_buffer.seek(0)
    return io_buffer

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

def streaming_generator(role, text):
    tts_generator = chatsystem.inference_end2end(text, '', '', stream=True, spk_id=role, this_uuid=this_uuid)
    yield wave_header_chunk()
    for out in tts_generator:
        chunk = out['tts_speech'].cpu().numpy()
        yield pack_audio(BytesIO(), chunk).getvalue()
        
def generate_audio(role, text):
    # role = "孙悟空"
    this_uuid = None
    pre_time = time.time()
    # set_all_random_seed(seed)
    for out in chatsystem.inference_end2end(text, '', '', stream=True, spk_id=role, this_uuid=this_uuid):
        chunk = out['tts_speech'].cpu().numpy()
        chunk = normalize_audio(chunk).flatten()
        cur_time = time.time()
        print(cur_time-pre_time)
        pre_time = cur_time
        yield sample_rate, chunk
    text = out['text_answer'][0] if isinstance(out['text_answer'], list) else out['text_answer']
    print(text)

        
        

def main():
    with gr.Blocks() as demo:
        gr.Markdown("#### 请选择对话角色或者自定义音色")

        with gr.Row():
            with gr.Accordion("选择角色", open=True):
                role = gr.Dropdown(choices=sft_spk)

            with gr.Accordion("自定义音色：请选择上传参考音频或者录制参考音频", open=False):
                with gr.Column():
                    with gr.Row():
                        prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
                        prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
                    prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')

            # with gr.Column():
            #     seed_button = gr.Button(value="\U0001F3B2")
            #     seed = gr.Number(value=0, label="随机推理种子")

        text = gr.Textbox(label="输入文本", lines=1, placeholder="请输入问题...")
        generate_button = gr.Button("进行提问")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        # seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[role, text],
                              outputs=[audio_output])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    args = parser.parse_args()

    sft_spk = chatsystem.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']

    main()