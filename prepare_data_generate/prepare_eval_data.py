import os
import json

data_root = "../../data/llm/llmanswer"
eval_txt = os.path.join(data_root, 'eval.txt')
lines = open(eval_txt).readlines()

save_name = os.path.join(data_root, "eval_question.txt")
f = open(save_name, 'w')

for l_id, l in enumerate(lines):
    data_name, utt = l.strip().split('\t')
    data_file = os.path.join(data_root, data_name, 'text',f"{utt}.json")
    text_info = json.load(open(data_file, 'r'))
    question = text_info['question']
    print(question)
    f.write(question+'\n')
f.close()
        