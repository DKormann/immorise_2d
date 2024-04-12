#%%
import torch 
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_tensor_type(torch.cuda.FloatTensor)
to_nearest_64 = lambda x: round(x/64) * 64
model_scale = 1.
max_seq_len = 20

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

class LatentAttentionBlock(nn.Module):
  """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values."""
  def __init__(self, num_dim):
    super().__init__()

    self.dim        = num_dim
    self.qk_dim     = self.dim//qk_dim_div
    self.v_dim      = num_dim
    self.expand_dim = num_dim * expand_factor
    

    self.norm       = nn.LayerNorm(self.dim, bias=False)
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
    
    self.norm       = nn.LayerNorm(self.dim, bias=False)
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


ntoks = 100

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Embedding(ntoks, residual_depth)
    self.layers = torch.nn.ModuleList([LatentAttentionBlock(residual_depth) for _ in range(num_blocks)])
    self.out = nn.Linear(residual_depth, ntoks)
  def forward(self, x):
    x = self.emb(x)
    for layer in self.layers:
      x = layer(x)
    return self.out(x)


def gen_data(n):
  def gen_sample():
    x = torch.randint(0, ntoks, (int(max_seq_len/2),))
    x_ = x[torch.randperm(x.shape[0])]
    # x_ = x
    return torch.cat([x, x_])
  x = torch.stack([gen_sample() for _ in range(n)])
  # y = torch.cat([torch.zeros(n, int(max_seq_len/2), dtype=torch.int64), x[:, int(max_seq_len/2):]], dim=1)
  y= x
  x = torch.cat([torch.zeros(n,1, dtype=torch.int64),x[:,:-1]], dim=1)
  return x ,y

x,y = gen_data(10)
x.shape, y.shape

plt.imshow(x.detach().cpu())
plt.show()
plt.imshow(y.detach().cpu())
# %%

model = Model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

#%%

def train_step():
  x,y = gen_data(100)
  opt.zero_grad()
  y_pred = model(x)
  
  loss = F.cross_entropy( y_pred[:,max_seq_len//2:].reshape(-1, ntoks),y[:,max_seq_len//2:].reshape(-1))
  loss.backward()
  opt.step()
  return loss.item()

#%%

opt.param_groups[0]['lr'] = 1e-4
epochs = 10_000
for e in range(epochs):
  try:loss = train_step()
  except KeyboardInterrupt: break
    
  print(f"\rEpoch {e+1}/{epochs} Loss: {loss}", end="")
  if (e+1) % (epochs//10)==0: print("")

#%%

x,y = gen_data(1)
y_pred = model(x).cpu()[0].detach()

plt.imshow(y_pred)
# plt.plot(y.cpu()[0])

