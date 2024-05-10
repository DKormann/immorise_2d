# %%
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from itertools import cycle

torch.device('cuda')

# %%

from utils.data import gen_data, display, max_points
from utils.latent_attention import LatentAttentionBlock

x,y,_ = gen_data(1,False,True,False)

model_scale = 1.
to_nearest_64 = lambda x: round(x/64) * 64
residual_depth = to_nearest_64(384 * np.log2(1.+model_scale))
num_blocks = round(8 * np.log2(1.+model_scale))

#%%

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(100, 384, scale_grad_by_freq=True)
    self.blocks = nn.ModuleList([LatentAttentionBlock(residual_depth) for _ in range(num_blocks)])
    self.norm = nn.LayerNorm(384)
    self.out = nn.Linear(384, 100, bias=False)
  
  def forward(self, q:torch.Tensor):
    q = self.emb(q)
    q = self.norm(q)
    for block in self.blocks:
      q = block(q)
    q = self.norm(q)
    q = self.out(q)
    return q

net = Model().cuda().train()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
#%%
train_data = cycle([gen_data(100,False,True,False) for _ in range(100)])
test_data = [gen_data(100,False,True,False) ]
#%%
for i in range(1000):
  x,y,_ = next(train_data)
  opt.zero_grad()
  out = net(x.cuda())
  loss = F.cross_entropy(out.reshape(-1, 100),y.flatten().cuda())
  loss.backward()
  opt.step()
  print('\r', loss.item(),end='')
  if (i+1) %20 == 0:
    print('test loss:', F.cross_entropy(net(test_data[0][0]).view(-1,100), test_data[0][1].view(-1)).item())
# %%

print(" ********* Inference *********")

x = torch.zeros(1,1).long()
for i in range(max_points):
  out = net(x)[-1].argmax(-1)
  print(out.shape)
  x = torch.cat([x, out[0].view(1,1)],1)

x
