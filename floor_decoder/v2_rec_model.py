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

"""if import utils fails change directory to parent folder """
from utils.data import gen_data, display, max_points
from utils.latent_attention import LatentAttentionBlock, LatentBlock
#%%
if __name__ == '__main__':
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
qk_dim_div = 8
expand_factor = 2

#%%

def init_weights(*size): return nn.Parameter(nn.init.xavier_uniform_(torch.empty(*size)))
#%%

class AttentionLayer(nn.Module):
  def __init__(self, num_dim=384, max_seq_len= max_points*2, **kwargs):
    super().__init__()
    self.dim        = num_dim
    self.qk_dim     = self.dim // qk_dim_div
    self.expand_dim = num_dim * expand_factor
    self.expand     = init_weights(self.dim,2*self.qk_dim+2*self.expand_dim)
    self.project    = init_weights(self.expand_dim, self.dim)
    with torch.no_grad():
      self.position_bias_base = torch.arange(0, max_seq_len).unsqueeze(0).float() - torch.arange(0, max_seq_len).unsqueeze(1)
      self.negative_infinity_matrix_base = torch.empty_like(self.position_bias_base).fill_(-float("inf"))
      self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
    self.position_bias_mult = nn.Parameter(torch.tensor(4.))
  def get_mask(self, n): return torch.where(self.causal_mask[:n, :n], (self.position_bias_base[:n,:n] + self.position_bias_mult).sigmoid(), self.negative_infinity_matrix_base[:n, :n])
  def forward(self, x):
    nx = F.layer_norm(x, [self.dim])
    q, k, linear, pre_gelu = (nx @ self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), -1)
    local, v = (linear * F.gelu(pre_gelu)).split((self.expand_dim-self.dim, self.dim), -1)
    scores = torch.softmax((q @ k.transpose(-2, -1) / q.size(-1)**.5) + self.get_mask(x.shape[1]), -1)
    return x + torch.cat([local, scores @ v], -1) @ self.project

layer  = AttentionLayer()
assert layer(torch.randn(1, max_points*2, residual_depth)).shape == torch.Size([1, max_points*2, residual_depth])

class CrossAttention(AttentionLayer):
  def __init__(self, num_dim=384, max_seq_len=max_points*2):
    super().__init__(num_dim, max_seq_len)
    self.expand_kv = init_weights(self.dim, self.qk_dim + 2*self.dim)
    self.expand_q = init_weights(self.dim, self.qk_dim + 2*(self.expand_dim-self.dim))
  def forward(self, kv, x):
    kv = F.layer_norm(kv, [self.dim])
    q = F.layer_norm(x, [self.dim])
    k, linear_v, pre_gelu_v = (kv @ self.expand_kv).split((self.qk_dim, self.dim, self.dim), -1)
    q, linear_loc, pre_gelu_loc = (q @ self.expand_q).split((self.qk_dim, self.expand_dim-self.dim, self.expand_dim-self.dim), -1)
    v = linear_v * F.gelu(pre_gelu_v)
    loc = linear_loc * F.gelu(pre_gelu_loc)
    scores = torch.softmax((q @ k.transpose(-2, -1) / q.size(-1)**.5) + self.get_mask(q.shape[1]), -1)
    return x + torch.cat([loc, scores @ v], -1) @ self.project

clayer = CrossAttention()
clayer(torch.randn(1, max_points*2, residual_depth), torch.randn(1, max_points*2, residual_depth))    

class Model(nn.Module):
  def __init__(self, max_seq_len=max_points*2):
    super().__init__()
    self.freqs = torch.exp(torch.arange(0, residual_depth, 2).float() * -(9. / residual_depth))
    self.blocks = nn.Sequential(*[AttentionLayer(residual_depth,max_seq_len=max_seq_len) for _ in range(num_blocks)])
    self.out = nn.Linear(residual_depth, 100, bias=False)
    self.norm = nn.LayerNorm(residual_depth)
  
  def forward(self, q:torch.Tensor):
    q = self.freqs * q.unsqueeze(-1)
    q = torch.cat([torch.sin(q), torch.cos(q)], -1)
    return self.out(self.norm(self.blocks(q)))

net = Model()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
assert net(torch.zeros(1, max_points*2)).shape == torch.Size([1, max_points*2, 100])
_=[nn.init.xavier_uniform_(p) for p in net.parameters() if p.dim()>1]

#%%
base_blocks = num_blocks//2

class RModel(nn.Module):
  def __init__(self,max_seq_len=max_points*2):
    super().__init__()
    self.freqs = torch.exp(torch.arange(0, residual_depth, 2).float() * -(9. / residual_depth))
    self.base = nn.Sequential(*[AttentionLayer(residual_depth,max_seq_len) for _ in range(base_blocks)])
    self.cross = CrossAttention(residual_depth,max_seq_len)
    self.out = nn.Linear(residual_depth, 100, bias=False)
    
  def forward(self, x:torch.Tensor):
    x = self.freqs * x.unsqueeze(-1)
    x = torch.cat([torch.sin(x), torch.cos(x)], -1)
    
    kv = self.base(x)
    for i in range(2):
      x = self.cross(kv, x)
    x = F.layer_norm(x, [residual_depth])
    return self.out(x)

net = RModel()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
assert net(torch.zeros(1, max_points*2)).shape == torch.Size([1, max_points*2, 100])
_=[nn.init.xavier_uniform_(p) for p in net.parameters() if p.dim()>1]
#%%
epochs = 2000

if __name__ == '__main__':
  scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=n_batches, epochs=1000, pct_start=0.1)
  try:
    for i in range(epochs):
      x,y,_ = next(train_data)
      opt.zero_grad()
      loss = F.cross_entropy(net(x.cuda()).reshape(-1, 100),y.flatten().cuda())
      loss.backward()
      opt.step()
      print('\r', loss.item(),end='')
      if (i+1) %(epochs//10) == 0:
        print('test loss:', F.cross_entropy(net(test_data[0][0]).view(-1,100), test_data[0][1].view(-1)).item())
      scheduler.step()
  except KeyboardInterrupt: pass

# %%
if __name__ == '__main__':
  print(" ********* Inference *********")
  for i in range(10):
    x = torch.zeros(1,1).long()
    for i in range(max_points*2):
      out = net(x)[0,-1:]
      out = torch.multinomial(F.softmax(out, dim=-1), 1)
      x = torch.cat([x, out[0].view(1,1)],1)

    d = x[0,1:-1]
    d = d.where(d != 0, torch.tensor(np.nan)).cpu()
    if d.shape[0] % 2 == 1: d = d[:-1]
    d = d.view(-1,2).T
    plt.plot(d[0],d[1])
    plt.show()
