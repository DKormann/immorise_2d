#%%
import torch
try:from utils.data import gen_data
except: raise ImportError("not running from project root?")
from itertools import cycle

#%%
batchsize = 100
n_batches = 100
train_data = cycle([gen_data(batchsize) for _ in range(n_batches)])
#%%

class PointTransformer(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.w = torch
  

net = PointTransformer()


