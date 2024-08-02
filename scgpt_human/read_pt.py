import torch

# 假设 best_model.pt 是你的模型文件
model_file = 'best_model.pt'

# 加载模型参数
state_dict = torch.load(model_file)

# 打印所有参数的名称和形状
for param_tensor in state_dict:
    print(f"Parameter: {param_tensor}\t Shape: {state_dict[param_tensor].size()}")
