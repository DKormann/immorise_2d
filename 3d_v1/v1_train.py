#%%
from models.latent_fusion_transformer import LatentAttentionBlock, LatentCrossAttentionBlock
import torch
from torch import nn
from torch.nn import functional as F
from utils.data import get_floors


#%%

class Model(torch.Module):
  def __init__(self):
    super().__init__()

model = Model()

#%%

train_floors, val_floors, test_floors = get_floors()

#%%

tok_floors = []
for floor in train_floors:
  floor = torch.tensor(floor).view(-1,2,2)
  print(floor.shape, floor.dtype)
  break
