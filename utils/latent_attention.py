#%%
import torch 
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math

device = torch.device('cuda')
dtype = torch.float32

to_nearest_64 = lambda x: round(x/64) * 64

class LatentBlock(nn.Module):
  def __init__(self, num_dim, model_scale=1.0, max_seq_len=100, ):
    super().__init__()
    self.model_scale = model_scale
    self.max_seq_len = max_seq_len
    # self.n_toks = 100

    self.qk_dim_div = 8
    self.expand_factor = 2
    self.residual_depth = to_nearest_64(384 * math.log2(1.+model_scale))

    self.num_blocks = round(8 * math.log2(1.+model_scale))
    self.causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()

    with torch.no_grad():
      bias_range = torch.arange(-max_seq_len+1, 1).to(device, dtype)
      self.position_bias_base = bias_range.unsqueeze(0) - bias_range.unsqueeze(1)
      self.negative_infinity_matrix_base = torch.empty_like(self.position_bias_base).fill_(-float("inf"))
      self.causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool))



#%%
class LatentAttentionBlock(LatentBlock):
  """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values."""
  def __init__(self, num_dim=384, **kwargs):
    super().__init__(num_dim, **kwargs)
    self.dim        = num_dim
    self.qk_dim     = self.dim // self.qk_dim_div
    self.v_dim      = num_dim
    self.expand_dim = num_dim * self.expand_factor

    self.norm       = nn.LayerNorm(self.dim)
    self.expand     = nn.Parameter(.5 * 1./self.residual_depth**.5 * 1./self.expand_factor * torch.randn(2*self.qk_dim+2*self.expand_dim, self.dim))
    self.project    = nn.Parameter(1. * 1./self.residual_depth**.5 * 1./self.expand_factor * 1./self.num_blocks * torch.randn((self.dim, self.expand_dim),dtype=dtype))
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
#%% 
if __name__ == '__main__':
  block = LatentAttentionBlock()
  x = torch.randn(100, 384)
  out = block(x)
