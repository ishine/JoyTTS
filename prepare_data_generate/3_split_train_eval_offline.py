import os
import json
import time
import torch
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def split_data(all_utt_list, f_count):
    ##获取已经存在的utt
    # utt_list, train_utt_list, eval_utt_list = cache_data(train_file, eval_file)
    utt_list, train_utt_list, eval_utt_list = [],[],[]
    data_count = len(utt_list)
    data_count_train = len(train_utt_list)
    data_count_eval = len(eval_utt_list)

    train_file = os.path.join(data_root, data_name, f"train_{f_count}.txt")
    f_train = open(train_file, 'w')
    if eval_ratio!=-1:
        eval_file = os.path.join(data_root, data_name, f"eval_{f_count}.txt")
        f_eval = open(eval_file, 'w')

    file_path = os.path.join(data_root, data_name)
    text_file_path = os.path.join(file_path, "text")
    # text_token_file_path = os.path.join(file_path, "text_token")
    hidden_states_file_path = os.path.join(file_path, "hidden_states")
    speech_token_file_path = os.path.join(file_path, "speech_token")
    # import ipdb; ipdb.set_trace()
    for utt in all_utt_list:
        
        # if utt in utt_list:
        #     continue
        speech_token_path_name = os.path.join(speech_token_file_path, "{}.pt".format(utt))
        text_path_name = os.path.join(text_file_path, "{}.json".format(utt))
        # text_token_path_name = os.path.join(text_token_file_path, "{}.pt".format(utt))
        hidden_states_path_name = os.path.join(hidden_states_file_path, "{}.pt".format(utt))
        if not os.path.exists(speech_token_path_name):
            continue
        # try:
        #     speech_token = torch.load(speech_token_path_name)
        # except:
        #     print(f"{speech_token_path_name} 打开失败！！！")
        #     continue
            
        data_count += 1
        # print(data_name+'\t'+utt+'\n')
        if eval_ratio == -1:
            f_train.write(data_name+'\t'+utt+'\n')
            data_count_train += 1
        else:
            if data_count % eval_ratio == 0:
                f_eval.write(data_name+'\t'+utt+'\n')
                data_count_eval += 1
            else:
                f_train.write(data_name+'\t'+utt+'\n')
                data_count_train += 1
    
            
    f_train.close()
    if eval_ratio!=-1:
        f_eval.close()
    return data_count, data_count_train, data_count_eval
        
def cache_data(train_file, eval_file):

    train_utt_list = []
    if os.path.exists(train_file):
        f_train_cache = open(train_file, 'r')
        for l_id, l in enumerate(f_train_cache):
            data_name_, utt = l.strip().split('\t')
            train_utt_list.append(utt)
        f_train_cache.close()
    
    eval_utt_list = []
    if os.path.exists(eval_file):
        f_eval_cache = open(eval_file, 'r')
        for l_id, l in enumerate(f_eval_cache):
            data_name_, utt = l.strip().split('\t')
            eval_utt_list.append(utt)
        f_eval_cache.close()
    
    utt_list = train_utt_list + eval_utt_list
    if len(utt_list) != len(set(utt_list)):
        import ipdb; ipdb.set_trace()
    return utt_list, train_utt_list, eval_utt_list

def read_datalist():
    datalist_path = os.path.join(data_root, data_name, "data.list")
    f = open(datalist_path, 'r')

    all_utt_list = [] #根据原始对话数据得到的utt列表
    for l_id, l in enumerate(f):
        utt = l.strip()
        all_utt_list.append(utt)
    return all_utt_list

def sort_merge_data():
    file_root = os.path.join(data_root, data_name)
    train_file_list = [os.path.join(file_root, file) for file in os.listdir(file_root) if file.startswith('train_') and file.endswith('.txt')]
    eval_file_list = [os.path.join(file_root, file) for file in os.listdir(file_root) if file.startswith('eval_') and file.endswith('.txt')]
    
    train_info_list = []
    for train_file in train_file_list:
        if os.path.exists(train_file):
            f_train_cache = open(train_file, 'r')
            for l_id, l in enumerate(f_train_cache):
                data_name_, utt = l.strip().split('\t')
                train_info_list.append({'utt':utt,'data_name':data_name_})     
            f_train_cache.close()
            os.system(f'rm {train_file}')
    
    eval_info_list = []
    for eval_file in eval_file_list:
        if os.path.exists(eval_file):
            f_eval_cache = open(eval_file, 'r')
            for l_id, l in enumerate(f_eval_cache):
                data_name_, utt = l.strip().split('\t')
                eval_info_list.append({'utt':utt,'data_name':data_name_})
            f_eval_cache.close()
            os.system(f'rm {eval_file}')
        
    train_info_list = sorted(train_info_list, key=lambda x: x['utt'])
    eval_info_list = sorted(eval_info_list, key=lambda x: x['utt'])

    f_train = open(os.path.join(file_root, "train.txt"), 'w')

    for train_info in train_info_list:
        data_name_ = train_info['data_name']
        utt = train_info['utt']
        f_train.write(data_name_+'\t'+utt+'\n')
    f_train.close()
    
    f_eval = open(os.path.join(file_root, "eval.txt"), 'w')
    for eval_info in eval_info_list:
        data_name_ = eval_info['data_name']
        utt = eval_info['utt']
        f_eval.write(data_name_+'\t'+utt+'\n')
    f_eval.close()
    
    data_count_train = len(train_info_list)
    data_count_eval = len(eval_info_list)
    data_count = data_count_train+data_count_eval
    print(f"数据总量{data_count}，其中训练集{data_count_train}，验证集{data_count_eval}")

executor = ThreadPoolExecutor(max_workers=16)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../../data/llm/")
    parser.add_argument('--data_name', type=str, default="RedGPT-main")
    parser.add_argument('--eval_ratio', type=int, default=10)
    parser.add_argument('--data_set_num', type=int, default=10000)
    args = parser.parse_args()

    data_root = os.path.join(args.data_root, 'llmanswer')
    data_name = args.data_name
    eval_ratio = args.eval_ratio
    data_set_num = args.data_set_num

    ##获取所有的utt
    all_utt_list = read_datalist()
    all_task = []

    
    for i, j in enumerate(range(0, len(all_utt_list), data_set_num)): 
        all_task.append(executor.submit(split_data, all_utt_list[j: j + data_set_num], i))
    data_count_all = 0 
    data_count_train_all = 0 
    data_count_eval_all = 0 
    for future in tqdm(as_completed(all_task)):
        data_count, data_count_train, data_count_eval = future.result()
        data_count_all += data_count
        data_count_train_all += data_count_train
        data_count_eval_all += data_count_eval
    print(f"数据总量{data_count_all}，其中训练集{data_count_train_all}，验证集{data_count_eval_all}")
    
    # split_data(all_utt_list[:data_set_num], 0)


    sort_merge_data()
    


    