#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle

dtype = torch.float32
if torch.cuda.is_available():
  device = torch.device('cuda')
  torch.set_default_device(device)
else:
  device = torch.device("mps")
  torch.set_default_device(device)


to_nearest_64 = lambda x: round(x/64) * 64
model_scale = 1.
max_seq_len = 100

qk_dim_div = 8
expand_factor = 2
residual_depth = to_nearest_64(384 * math.log2(1.+model_scale))

num_blocks = round(8 * math.log2(1.+model_scale))
causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
n_toks = 100

with torch.no_grad():
  bias_range = torch.arange(-max_seq_len+1, 1).to(device, dtype)
  position_bias_base = bias_range.unsqueeze(0) - bias_range.unsqueeze(1)
  negative_infinity_matrix_base = torch.empty_like(position_bias_base).fill_(-float("inf"))
  causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool))
#%%
class LatentAttentionBlock(nn.Module):
  """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values."""
  def __init__(self, num_dim):
    super().__init__()
    self.dim        = num_dim
    self.qk_dim     = self.dim//qk_dim_div
    self.v_dim      = num_dim
    self.expand_dim = num_dim * expand_factor
    

    self.norm       = nn.LayerNorm(self.dim)
    self.expand     = nn.Parameter(.5 * 1./residual_depth**.5 * 1./expand_factor * torch.randn(2*self.qk_dim+2*self.expand_dim, self.dim))
    self.project    = nn.Parameter(1. * 1./residual_depth**.5 * 1./expand_factor * 1./num_blocks * torch.randn((self.dim, self.expand_dim),dtype=dtype))
    self.position_bias_mult = nn.Parameter(torch.tensor(1.))

  def forward(self, x):
  
    residual = x
    attn_mask = torch.where(causal_mask[:x.shape[1], :x.shape[1]], F.softplus(self.position_bias_mult) * position_bias_base[:x.shape[1], :x.shape[1]], negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])
    x = self.norm(x)
    query, key, linear, pre_gelu = F.linear(x, self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)
    geglu = linear * F.gelu(pre_gelu)
    geglu_local, geglu_attention_value = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
    attention = F.scaled_dot_product_attention(query, key, geglu_attention_value, attn_mask=attn_mask)
    out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)
    x = residual + out
    return x
  
class LatentCrossAttentionBlock(nn.Module):
  def __init__ (self, num_dim):
    super().__init__()
    
    self.dim        = num_dim
    self.qk_dim     = self.dim//qk_dim_div
    self.v_dim      = num_dim
    self.expand_dim = num_dim * expand_factor
    self.local_dim  = self.expand_dim - self.v_dim
    
    self.norm       = nn.LayerNorm(self.dim)
    self.Wq         = nn.Parameter(.5 * 1./residual_depth**.5 * 1./expand_factor * torch.randn(self.qk_dim + 2 * self.local_dim, self.dim))
    self.Wkv        = nn.Parameter(.5 * 1./residual_depth**.5 * 1./expand_factor * torch.randn(self.qk_dim + 2 * self.v_dim, self.dim))
    
    self.project    = nn.Parameter(1. * 1./residual_depth**.5 * 1./expand_factor * 1./num_blocks * torch.randn((self.dim, self.expand_dim),dtype=dtype))

  def forward(self, q, kv):
    residual = q
    q = self.norm(q)
    query, lin_local, pre_geglu_local = F.linear(q, self.Wq).split((self.qk_dim, self.local_dim, self.local_dim), dim=-1)
    geglu_local = lin_local * F.gelu(pre_geglu_local)
    key, lin_value, pre_geglu_value = F.linear(kv, self.Wkv).split((self.qk_dim, self.v_dim, self.v_dim), dim=-1)
    geglu_value = lin_value * F.gelu(pre_geglu_value)
    attention = F.scaled_dot_product_attention(query, key, geglu_value)

    out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)
    return residual + out

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(n_toks, residual_depth, scale_grad_by_freq=True)
    self.pt_emb = nn.Embedding(n_toks, residual_depth//2, scale_grad_by_freq=True)
    self.blocks = nn.ModuleList([LatentAttentionBlock(residual_depth) for _ in range(num_blocks)])
    self.blocks2 = nn.ModuleList([LatentCrossAttentionBlock(residual_depth) for _ in range(num_blocks)])
    self.norm = nn.LayerNorm(residual_depth)
    self.out = nn.Linear(residual_depth, n_toks, bias=False)
  
  def forward(self, q:torch.Tensor, pts:torch.Tensor):
    q = self.emb(q)
    kv = self.pt_emb(pts) # B, 100, 2, D / 2
    kv = kv.view(kv.shape[0], -1, residual_depth) # B, 100 , D
    
    q = self.norm(q)
    for block, block2 in zip(self.blocks, self.blocks2):
      q = block(q)
      q = block2(q, kv)
    q = self.norm(q)
    q = self.out(q)
    return q

def step(x,y,pts):
  opt.zero_grad()
  out = net(x,pts)
  loss = F.cross_entropy(out.reshape(-1, n_toks),y.flatten())
  loss.backward()
  opt.step()
  return loss

def test():
  global test_data
  # for x, y, pts in val_data:
  #   with torch.no_grad():
  #     p = net(x,pts)
  #     err += F.cross_entropy(p.view(-1, n_toks), y.view(-1)).item()
  errs = [F.cross_entropy(net(x,pts).view(-1, n_toks), y.view(-1)).item() for x,y,pts in test_data]
  return sum(errs) / len(errs)

from torch.optim.lr_scheduler import StepLR

#%%

from utils.data import gen_data, display, max_points
#%%
batchsize = 100 
n_batches = 100
test_data = [gen_data(batchsize) for _ in range(n_batches)]
#%%
train_data = cycle([gen_data(batchsize) for _ in range(n_batches)])
#%%
net = Model().to(device, dtype).train()
opt = optim.Adam(net.parameters(), lr=1e-3)

epochs = 1000

#%%
opt = optim.Adam(net.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs, pct_start=0.1)

for e in range(epochs):
  x, y, pts = next(train_data)
  try: loss = step(x, y, pts)
  except KeyboardInterrupt:break
  print(f"\r{e+1}/{epochs}: ",loss.item(), end="")
  if (e+1) % (epochs // 10) == 0: 
    print(" val loss:", test())

  scheduler.step()


#%%

print ("******* INFERENCE ********")

def generate(n):
  x = torch.zeros(1, 1).long()
  _,_,pts = gen_data(1)
  for i in range(n):
    p = net(x[:,-99:], pts)
    choices = p[0,-1,:]
    # choice = choices.argmax(-1)
    choice = torch.multinomial(choices.softmax(-1),1)
    
    x = torch.cat([x, choice.reshape(1,1)], dim=1)
  x = x.where(x != 0, torch.tensor(np.nan)).cpu()
  return x,pts

for i in range(5):
  p,pts = generate(max_points*2)
  plt.scatter(pts[0,:,0].cpu().numpy(), pts[0,:,1].cpu().numpy())
  plt.plot(*p[0,1:].reshape(-1,2).T.cpu())
  plt.show()

# %%
