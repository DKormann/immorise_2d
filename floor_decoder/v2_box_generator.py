# %%
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from itertools import cycle

import math
from floor_decoder.data import dataset, floors

torch.set_default_device('cuda')
device = torch.device('cuda')
#%%


ds = []
rest = dataset
for floor in floors:
  rooms, rest = rest[:floor['rooms']], rest[floor['rooms']:]
  ds.append(rooms)

def boxit(room):
  x, y = room[:,0], room[:,1]
  return np.array([x.min(), y.min(), x.max(), y.max()])

def drawbox(box,**kwargs):
  x1, y1, x2, y2 = box
  plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],**kwargs)

#%%

for floor in ds:
  plt.axis('equal')
  for room in floor:
    drawbox(boxit(room), color='gray')
    # plt.plot(room[:,0], room[:,1],color='black')
  plt.show()

#%%

def area(box):return (box[2]-box[0])*(box[3]-box[1])

bxs = [torch.stack(sorted([torch.tensor(boxit(room)) for room in floor], key=area, reverse=True)) for floor in ds]
maxlen = max(len(f) for f in bxs)

bxs = torch.stack([torch.cat([f, torch.zeros(maxlen - len(f), 4)]) for f in bxs])
split = int(len(bxs) * .9)
train_bxs, test_bxs = bxs[:split], bxs[split:]



