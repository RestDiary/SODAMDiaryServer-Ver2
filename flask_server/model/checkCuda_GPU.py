# import numpy as np
# import torch
# from torch.utils.data import dataloader
# from tqdm import tqdm
# from transformers import AdamW
# ctx = "cuda" if torch.cuda.is_available() else "cpu"
# print(ctx)

import torch

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
