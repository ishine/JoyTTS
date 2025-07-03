
import json
import requests
from time import time
import os
import sys
import argparse
from tqdm import tqdm

def split_data(data_file, eval_ratio):

    
    data = json.load(open(data_file, 'r'))
    data_num = len(data)
    print(f"get data number {data_num}")
    

    train_file = os.path.join(data_root, data_name, f"train.json")
    f_train = open(train_file, 'w')
    if eval_ratio!=-1:
        eval_file = os.path.join(data_root, data_name, f"eval.json")
        f_eval = open(eval_file, 'w')

    data_train = []
    data_eval = []
    # import ipdb; ipdb.set_trace()
    for id in tqdm(range(data_num)):
    # for id in tqdm(range(5)):
        data_i = data[id]
        
        if eval_ratio == -1:
            data_train.append(data_i)
        else:
            if id % eval_ratio == 0:
                data_eval.append(data_i)
            else:
                data_train.append(data_i)
    
    f = open(train_file, 'w')
    json.dump(data_train, f, ensure_ascii=False, indent=4)
    f.close()
        
    if eval_ratio!=-1:

        f = open(eval_file, 'w')
        json.dump(data_eval, f, ensure_ascii=False, indent=4)
        f.close()

    data_count_train = len(data_train)
    data_count_eval = len(data_eval)
    data_count = data_count_train+data_count_eval
    print(f"数据总量{data_count}，其中训练集{data_count_train}，验证集{data_count_eval}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../../data/llm/")
    parser.add_argument('--data_name', type=str, default="RedGPT-main")
    args = parser.parse_args()
    
    data_root = args.data_root
    data_name = args.data_name
    
    data_file = os.path.join(data_root, data_name, "normalized_data.json")

    split_data(data_file, eval_ratio=50)