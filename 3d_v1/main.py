#%% 
import math
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from utils.data import get_floors, display_floor
from models.latent_fusion_transformer import LatentAttentionBlock ,to_nearest_64#, LatentCrossAttentionBlock

#%%

train, test, val = get_floors()

#%%
train = sorted(train, key=len)
maxlen = len(train[-1])
split_idxs = [52,65]
train_batches = [train[:52], train[52:65], train[65:]]
for i, batch in enumerate(train_batches):
  print(f"Batch {i} has {len(batch)} floors")
  maxl = max([len(floor) for floor in batch])
  batch = torch.stack([F.pad(floor, (0,0,0,0,0,maxl-len(floor))) for floor in batch])
  train_batches[i] = batch.view(*batch.shape[:-3], -1)
  

#%%
model_scale = 1.
class Model(nn.Module):
  def __init__(self, n_toks=100):
    super().__init__()
    num_blocks = round(8 * math.log2(1.+model_scale))
    self.residual_depth = to_nearest_64(384 * math.log2(1.+model_scale))
    self.emb = nn.Embedding(n_toks, self.residual_depth, scale_grad_by_freq=True)
    self.blocks = [LatentAttentionBlock(self.residual_depth, max_seq_len=maxlen) for _ in range(num_blocks)]
    self.norm = nn.LayerNorm(self.residual_depth, bias=False)
    self.out = nn.Linear(self.residual_depth, n_toks, bias=False)
  
  def forward(self,x):
    x = self.emb(x)
    for block in self.blocks: x = block(x)
    return self.out(x)

net = Model()

p = net(train_batches[0])
p.shape