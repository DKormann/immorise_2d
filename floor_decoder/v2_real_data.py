# %%
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from itertools import cycle
from utils.latent_attention import LatentBlock
import math
from floor_decoder.data import dataset, floors

torch.set_default_device('cuda')
device = torch.device('cuda')

#%%

def show_room(f):
  f = f.view(-1, 2)
  plt.plot(f[:,0], f[:,1])
  plt.axis('equal')
  plt.show()

#%%

sorted_dataset = sorted(dataset, key=lambda f: len(f))
plt.scatter([*range(len(dataset))], [len(f) for f in sorted_dataset])
sum(len(f) for f in sorted_dataset)
#%%

ds = sorted_dataset[1000:-100]
ds = [torch.cat([torch.tensor(f), torch.tensor(f[0:1])]) for f in ds]

maxlen = max(len(f) for f in ds)
minlen = min(len(f) for f in ds)
maxlen,minlen, sum(len(f) for f in ds)
print(maxlen,minlen, sum(len(f) for f in ds))

def show_room(f):
  f = f.view(-1, 2)
  f = f.where(f != 0, torch.tensor(torch.nan)).cpu()
  plt.plot(f[:,0], f[:,1])
  plt.axis('equal')
  plt.show()

def discretize(f, n=100):
  f = f - f.min(0, keepdim=True).values
  f = f / f.max() * (n-1)
  f = f.round().long() + 1
  f = f.clamp(1, n-1).view(-1)
  f = torch.cat([f, torch.zeros(maxlen*2 - len(f)).long()])
  return f

ds = torch.stack([discretize(f) for f in ds])
split = int(len(ds) * .8)
train_ds, test_ds = ds[:split], ds[split:]
#%%

model_scale = 1.
to_nearest_64 = lambda x: round(x/64) * 64
residual_depth = to_nearest_64(384 * np.log2(1.+model_scale))
num_blocks = round(8 * np.log2(1.+model_scale))

max_points = maxlen*2
position_bias_base = torch.zeros(max_points*2, max_points*2)
inner_boost = torch.eye(max_points*2,max_points*2+1)[:,1:]
position_bias_base[::2] += inner_boost[::2] * 1.5

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
batch_size = 100
n_batches = len(ds) / batch_size
net = Model().cuda().train()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=n_batches, epochs=1000, pct_start=0.1)
sum(np.prod(p.shape) for p in net.parameters())



#%%

def step(x):
  y = x
  x = torch.cat([torch.zeros_like(y[:,0:1]), y[:,:-1]], 1)
  p = net(x)
  return F.cross_entropy(p.view(-1,100), y.view(-1))

epochs = 1000
for e in range(epochs):
  for k in range(0,len(ds),batch_size):
    batch = ds[k:k+batch_size]
    loss = step(batch)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print('\r', loss.item(),end='')
  val = step(test_ds)
  if (e+1) % (epochs//20) == 0: print("eval", val.item())
  
#%%
print(" ********* Inference *********")
for i in range(10):
  x = torch.zeros(1,1).long()
  for j in range(max_points):
    p = net(x)
    p = p[:,-1]
    
    choice = torch.multinomial(F.softmax(p, -1), 1)
    x = torch.cat([x, choice.view(1,1)], 1)
    
  show_room(x[0,1:])