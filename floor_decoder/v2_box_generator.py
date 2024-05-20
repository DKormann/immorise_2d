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

def area(box):return (box[2]-box[0])*(box[3]-box[1])

bxs = [torch.stack(sorted([torch.tensor(boxit(room)) for room in floor], key=area, reverse=True)) for floor in ds]
for floor in bxs:
  floor[:,::2] -= floor[:,::2].min()
  floor[:,1::2] -= floor[:,1::2].min()
  floor[:] = floor[:] / floor.max() * 99 + 1

bxs = [floor.long().clamp(1,99).reshape(-1) for floor in bxs]

#%%
maxlen = max(len(f) for f in bxs)
bxs = torch.stack([torch.cat([f, torch.zeros(maxlen - len(f)).long()]) for f in bxs])

split = int(len(bxs) * .9)
train_bxs, test_bxs = bxs[:split], bxs[split:]

# %%
for floor in train_bxs[:]:
  plt.xlim(0,100)
  plt.ylim(0,100)
  
  for room in floor.view(-1,4):    
    drawbox(room.cpu(), color='gray')
  plt.show()
  break

#%%
from floor_decoder.v2_model import Model
model = Model(group_size=1, max_seq_len=train_bxs.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

#%%

y = train_bxs
x = torch.cat([torch.zeros(y.shape[0],1), y[:,:-1]],1)
# %%
epochs = 1000
for e in range(epochs):
  p = model(x)
  loss = F.cross_entropy(p.view(-1,100), y.view(-1))
  opt.zero_grad()
  loss.backward()
  opt.step()
  print('\r', loss.item(), end='')
  
# %%
print(" ********* Inference *********")
for i in range(10):
  x = torch.zeros(1,1).long()
  for i in range(maxlen):
    out = model(x)[0,-1:]
    out = torch.multinomial(F.softmax(out, dim=-1), 1)
    x = torch.cat([x, out[0].view(1,1)],1)
  
  x = x[0,1:]
  x = x[:(x.argmin()//4) * 4]
  plt.xlim(0,100)
  plt.ylim(0,100)
  for box in x.view(-1,4):
    drawbox(box.cpu(), color='gray')
  break

#%%
x[0][1:].argmin()