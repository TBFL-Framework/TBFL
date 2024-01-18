import torch

def set_seed(cfg):
    torch.manual_seed(cfg.torch_seed)

    print('Seed finished...')