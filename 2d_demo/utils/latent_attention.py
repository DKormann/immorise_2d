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
# model_scale = 1.
# max_seq_len = 100

# qk_dim_div = 8
# expand_factor = 2
# residual_depth = to_nearest_64(384 * math.log2(1.+model_scale))

# num_blocks = round(8 * math.log2(1.+model_scale))
# causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()

# with torch.no_grad():
#   bias_range = torch.arange(-max_seq_len+1, 1).to(device, dtype)
#   position_bias_base = bias_range.unsqueeze(0) - bias_range.unsqueeze(1)
#   negative_infinity_matrix_base = torch.empty_like(position_bias_base).fill_(-float("inf"))
#   causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool))

#%%
class LatentAttentionBlock(nn.Module):
  """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values."""
  def __init__(self,  num_dim, model_scale = 1., max_seq_len = 100):
    self.model_scale = model_scale
    self.max_seq_len = max_seq_len
    
    qk_dim_div = 8
    expand_factor = 2
    residual_depth = to_nearest_64(384 * math.log2(1.+model_scale))
    num_blocks = round(8 * math.log2(1.+model_scale))
    self.causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()

    with torch.no_grad():
      bias_range = torch.arange(-max_seq_len+1, 1).to(device, dtype)
      self.position_bias_base = bias_range.unsqueeze(0) - bias_range.unsqueeze(1)
      self.negative_infinity_matrix_base = torch.empty_like(self.position_bias_base).fill_(-float("inf"))
      self.causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool))

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
    attn_mask = torch.where(self.causal_mask[:x.shape[1], :x.shape[1]], F.softplus(self.position_bias_mult) * self.position_bias_base[:x.shape[1], :x.shape[1]], self.negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])
    x = self.norm(x)
    query, key, linear, pre_gelu = F.linear(x, self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)
    geglu = linear * F.gelu(pre_gelu)
    geglu_local, geglu_attention_value = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
    attention = F.scaled_dot_product_attention(query, key, geglu_attention_value, attn_mask=attn_mask)
    out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)
    x = residual + out
    return x
  
# class LatentCrossAttentionBlock(LatentAttentionBlock):
#   def __init__ (self, num_dim, model_scale = 1., max_seq_len = 100):
#     super().__init__(num_dim, model_scale, max_seq_len)
    

#     self.Wq         = nn.Parameter(.5 * 1./residual_depth**.5 * 1./expand_factor * torch.randn(self.qk_dim + 2 * self.local_dim, self.dim))
#     self.Wkv        = nn.Parameter(.5 * 1./residual_depth**.5 * 1./expand_factor * torch.randn(self.qk_dim + 2 * self.v_dim, self.dim))
    
#     self.project    = nn.Parameter(1. * 1./residual_depth**.5 * 1./expand_factor * 1./num_blocks * torch.randn((self.dim, self.expand_dim),dtype=dtype))

#   def forward(self, q, kv):
#     residual = q
#     q = self.norm(q)
#     query, lin_local, pre_geglu_local = F.linear(q, self.Wq).split((self.qk_dim, self.local_dim, self.local_dim), dim=-1)
#     geglu_local = lin_local * F.gelu(pre_geglu_local)
#     key, lin_value, pre_geglu_value = F.linear(kv, self.Wkv).split((self.qk_dim, self.v_dim, self.v_dim), dim=-1)
#     geglu_value = lin_value * F.gelu(pre_geglu_value)
#     attention = F.scaled_dot_product_attention(query, key, geglu_value)

#     out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)
#     return residual + out    
