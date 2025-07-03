import torch
import os 

now_dir = os.getcwd()

model_path = os.path.join(now_dir, "../../exp/cosyvoice2/llm/deepspeed/init")
rank_dirs = ["path_to_rank_0", "path_to_rank_1", ...]
final_state_dict = {}

for rank_dir in rank_dirs:
    # 加载每个rank的checkpoint
    checkpoint = torch.load(os.path.join(rank_dir, "checkpoint.pt"))
    model_state_dict = checkpoint["model"]  # 假设模型状态保存在"model"键下
    
    # 合并state_dicts
    for key in model_state_dict:
        if key not in final_state_dict:
            final_state_dict[key] = model_state_dict[key]
        else:
            # 根据实际情况选择如何合并，例如求平均或简单累加等
            final_state_dict[key] += model_state_dict[key]  # 示例操作，根据需要调整

# 现在final_state_dict包含了所有rank的模型参数
