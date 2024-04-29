#%% 
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from utils.data import get_floors, display_floor
from models.latent_fusion_transformer import LatentAttentionBlock, LatentCrossAttentionBlock

#%%

train, test, val = get_floors()

#%%
train = sorted(train, key=len)
maxlen = len(train[-1])
split_idxs = [52,65]
train_batches = [train[:52], train[52:65], train[65:]]
#%%
for i, batch in enumerate(train_batches):
  print(f"Batch {i} has {len(batch)} floors")
  maxl = max([len(floor) for floor in batch])
  batch = torch.stack([F.pad(floor, (0,0,0,0,0,maxl-len(floor))) for floor in batch])
  train_batches[i] = batch
#%%

