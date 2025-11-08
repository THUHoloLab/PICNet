import os
import torch
import random
import numpy as np

def generate_zernike(batch_size=1, device='cpu'):
    z = torch.rand(batch_size, 12, device=device) - 0.5
    z[:, [1, 9]] = torch.rand(batch_size, 2, device=device) * 3 - 1.5
    z[:, [0, 2]] = torch.rand(batch_size,2,device=device)*2 - 1
    return z


def set_seed(seed=None):
    random.seed(seed)                   # Python random模块
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # PyTorch CPU
    torch.cuda.manual_seed(seed)       # PyTorch GPU
    torch.cuda.manual_seed_all(seed)   # 所有GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def center_crop(H, size):
    batch, channel, Nh, Nw = H.size()

    return H[:, :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]