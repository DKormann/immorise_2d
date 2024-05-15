# %%
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from itertools import cycle

import math

torch.set_default_device('cuda')
device = torch.device('cuda')

# if import utils fails change directory to parent folder 
from utils.data import gen_data, display, max_points
from utils.latent_attention import LatentAttentionBlock, LatentBlock
#%%
def genfn(n):return gen_data(n, False,True)
batch_size = 100
n_batches = 200
train_data = cycle([genfn(batch_size) for _ in range(n_batches)])
test_data = [genfn(100)]

#%%
x,y,_ = gen_data(1,False,True,False)

model_scale = 1.
to_nearest_64 = lambda x: round(x/64) * 64
residual_depth = to_nearest_64(384 * np.log2(1.+model_scale))
num_blocks = round(8 * np.log2(1.+model_scale))

#%%

position_bias_base = torch.zeros(max_points*2, max_points*2)
inner_boost = torch.eye(max_points*2,max_points*2+1)[:,1:]
position_bias_base[::2] += inner_boost[::2] * 1.5

plt.imshow(position_bias_base.cpu())
#%%
class AttentionLayer(LatentBlock):
  """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values."""
  def __init__(self, num_dim=384, **kwargs):
    super().__init__(num_dim, **kwargs)
    self.dim        = num_dim
    self.qk_dim     = self.dim // self.qk_dim_div
    self.v_dim      = num_dim
    self.expand_dim = num_dim * self.expand_factor

    self.norm       = nn.LayerNorm(self.dim)
    self.expand     = nn.Parameter(.5 * 1./self.residual_depth**.5 * 1./self.expand_factor * torch.randn(2*self.qk_dim+2*self.expand_dim, self.dim))
    self.project    = nn.Parameter(1. * 1./self.residual_depth**.5 * 1./self.expand_factor * 1./self.num_blocks * torch.randn((self.dim, self.expand_dim)))
    self.position_bias_base = position_bias_base

  def forward(self, x):
  
    residual = x
    attn_mask = torch.where(self.causal_mask[:x.shape[1], :x.shape[1]], self.position_bias_base[:x.shape[1], :x.shape[1]], self.negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])
    x = self.norm(x)
    query, key, linear, pre_gelu = F.linear(x, self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)
    geglu = linear * F.gelu(pre_gelu)
    geglu_local, geglu_attention_value = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
    attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1)
    attention = attn_weight @ geglu_attention_value

    out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)
    x = residual + out
    return x

#%%

class Enc(nn.Module):
  def __init__(self):
    super().__init__()

    self.freqs = torch.exp(torch.arange(0, residual_depth, 2).float() * -(math.log(10000.0) / residual_depth))

  def forward(self, q:torch.Tensor): 
    q = q.unsqueeze(-1) * self.freqs
    q = torch.cat([torch.sin(q), torch.cos(q)], -1)
    return q

#%%
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = Enc()
    self.blocks = nn.ModuleList([AttentionLayer(residual_depth) for _ in range(num_blocks)])
    self.norm = nn.LayerNorm(residual_depth)
    self.out = nn.Linear(residual_depth, 100, bias=False)
  
  def forward(self, q:torch.Tensor):
    q = self.emb(q)
    q = self.norm(q)
    for block in self.blocks:
      q = block(q)
    q = self.norm(q)
    q = self.out(q)
    return q
# %%
net = Model().cuda().train()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=n_batches, epochs=1000, pct_start=0.1)
sum(np.prod(p.shape) for p in net.parameters())

#%%


epochs = 2000

for i in range(epochs):
  x,y,_ = next(train_data)
  opt.zero_grad()
  out = net(x.cuda())
  loss = F.cross_entropy(out.reshape(-1, 100),y.flatten().cuda())
  loss.backward()
  opt.step()
  print('\r', loss.item(),end='')
  if (i+1) %(epochs//10) == 0:
    print('test loss:', F.cross_entropy(net(test_data[0][0]).view(-1,100), test_data[0][1].view(-1)).item())
  scheduler.step()
# %%

print(" ********* Inference *********")
for i in range(10):
  x = torch.zeros(1,1).long()
  for i in range(max_points*2):
    out = net(x)[0,-1:]
    out = torch.multinomial(F.softmax(out, dim=-1), 1)
    x = torch.cat([x, out[0].view(1,1)],1)

  d = x[0,1:-1]
  d = d.where(d != 0, torch.tensor(np.nan)).cpu()
  if d.shape[0] % 2 == 1:
    d = d[:-1]
  d = d.view(-1,2).T
  plt.plot(d[0],d[1])
  plt.show()

# %%
torch.randn(10,20).amax(-2).shape