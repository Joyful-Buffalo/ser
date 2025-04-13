import torch

# 加载模型
model = torch.load('/home/pwy/lhc/EMODB/trained_model/FullInception3IMPRO_IEMOCAP2023-05-29 00:48:26.pt',
                   map_location=torch.device('cpu'))

# 遍历模型的参数
for name, param in model.named_parameters():
    print(f'Parameter name: {name}')
    print(f'Parameter shape: {param.shape}')
    print(f'Parameter values: {param}')
    print(f'Parameter size (in bytes): {param.numel() * param.element_size()}')
    print('-------------------------')
