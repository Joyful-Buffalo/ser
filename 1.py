import torch

# 将张量移动到CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.randn(1000, 1000).to(device)

# 查看已分配的内存量
allocated_memory = torch.cuda.memory_allocated(device=device)
print(f"已分配的内存量: {allocated_memory} bytes", allocated_memory / 1e6)
