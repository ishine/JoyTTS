import json
data_name = "RedGPT-main"
ori_data_file = 'RedGPT-Dataset-V1-CN.json'
save_file = 'normalized_data.json'
f = open(ori_data_file, 'r')
lines = f.readlines()
f.close()
data_dict = {}
data = []
data_count = 0
for line in lines:
    data_i = json.loads(line)
    reference = data_i["reference"]
    dialogue = data_i["dialogue"]
    dialogue = dialogue.replace('：',":").replace(': ',":").replace('HUMAN',"Human").replace('Human(用户)',"Human")
    conversation_list = dialogue.split("Human:")
    conversations = []
    for conversation in conversation_list:
        if conversation == "":
            continue
        conversation = conversation.split("Assistant:")
        if len(conversation) != 2:
            print(f"data error {conversation}, skip")
            continue
        for id, text in enumerate(conversation):
            while text.startswith('\n'):
                text = text[2:]
            while text.endswith('\n'):
                text = text[:-2]
            conversation[id] = text

        conversation = [
                        {"role":"user", "content":conversation[0]},
                        {"role":"assistant", "content":conversation[1]}
                    ]
        conversations.extend(conversation)
    if len(conversations) > 0:
        
        data_id='%06d'%data_count
        utt = f"{data_name}_{data_id}"
        
        data.append({"reference":reference, "conversations":conversations, "utt":utt})
        # data.append({"reference":reference, "conversations":conversations})
        data_count += 1
print(f'收集多轮对话数据{len(data)}组,示例如下:')
print(data[0])
data_dict = {"data":data}
f = open(save_file, 'w')
json.dump(data, f, ensure_ascii=False, indent=4)
f.close()



