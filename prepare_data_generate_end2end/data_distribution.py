import os
import json
from tqdm import tqdm
import random

def read_datalist(datalist_path):
    f = open(datalist_path, 'r')

    all_utt_list = [] #根据原始对话数据得到的utt列表
    for l_id, l in enumerate(f):
        utt = l.strip()
        all_utt_list.append(utt)
    return all_utt_list
    
data_name_list = ["RedGPT-main", "generated_chat_0.4M"]
use_llmlabel=False

gen_save_file = "llmanswer_uselabel" if use_llmlabel else "llmanswer"
# data_name = "generated_chat_0.4M_augment"
data_root = "/media/cfs/zhoufangru/workspace/agent/data/llm/"

text_token_len_dict = {}
for data_name in data_name_list:
    datalist_path = os.path.join(data_root, gen_save_file, data_name, "data.list")
    
    all_utt_list = read_datalist(datalist_path)
    
    for llm_spk_id in tqdm(random.sample(all_utt_list,1000)):
        llm_json_name = llm_spk_id+".json"
        text_json_path = os.path.join(data_root, gen_save_file, data_name, "text", llm_json_name)
        if not os.path.exists(text_json_path):
            # print(f"text_json_path: {text_json_path} 不存在")
            continue
        ##读取本地的llm生成的tts_answer
        text_token = json.load(open(text_json_path, 'r'))['text_token']
        text = json.load(open(text_json_path, 'r'))['tts_input'][0]

        if len(text_token)>100:
            print(len(text), len(text_token), text)
        if len(text_token) in text_token_len_dict:
            text_token_len_dict[len(text_token)] += 1
        else:
            text_token_len_dict[len(text_token)] = 1
text_token_len_dict = sorted(text_token_len_dict.items(), key=lambda x:x[0])
print(text_token_len_dict)