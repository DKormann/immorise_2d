#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_tensor_type(torch.cuda.FloatTensor)
to_nearest_64 = lambda x: round(x/64) * 64
model_scale = 1.
max_seq_len = 100

qk_dim_div = 8
expand_factor = 2
residual_depth = to_nearest_64(384 * math.log2(1.+model_scale))

num_blocks = round(8 * math.log2(1.+model_scale))
causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()

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
   
n_toks = 100

#%%
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(n_toks, residual_depth, scale_grad_by_freq=True)
    self.blocks = nn.ModuleList([LatentAttentionBlock(residual_depth) for _ in range(num_blocks)])
    self.norm = nn.LayerNorm(residual_depth, bias=False)
    self.out = nn.Linear(residual_depth, n_toks, bias=False)
  
  def forward(self, x:torch.Tensor):
    x = self.emb(x)
    x = self.norm(x)
    for block in self.blocks: x = block(x)
    x = self.norm(x)
    x = self.out(x)
    return x

net = Model().to(device, dtype).train()
opt = optim.Adam(net.parameters(), lr=3e-4)
#%% gen data

def random_shape():
  n = np.random.randint(3, 10)
  corners = []
  for i in range(n):
    angle = 2*np.pi*i/n + np.random.rand()*0.1
    r = np.random.rand() * 0.5 + 0.5
    corners.append(torch.tensor([np.cos(angle)*r, np.sin(angle)*r],dtype=torch.float32))
  verts = []
  corner_dir = np.random.randint(2)
  for i in range(n + 1):
    verts.append(corners[i % n])
    verts.append(torch.tensor([corners[(i+corner_dir) % n][0] , corners[(i + 1 - corner_dir) % n][1]]))

  verts = torch.stack(verts[:-1])
  return verts

max_points = 19

def display(s):
  s = s[:2 * max_points]
  s = s.reshape(-1,2)
  plt.plot(s[:,0].cpu().numpy(), s[:,1].cpu().numpy())


def gen_data(n):
  shapes = [random_shape() for _ in range(n)]
  x = torch.stack([torch.cat([s, torch.zeros(max_points - len(s), 2)]) for s in shapes])

  x = x.view(n, -1)
  x = (x - x.min()) / (x.max() - x.min()) * (n_toks-1)
  x = x.long()
  pad = torch.zeros(n, 1).long()
  y = x
  x = torch.cat([pad, x], dim=1)[:, :-1]
  return x, y

x,y = gen_data(100)

#%%

def step(x,y):
  opt.zero_grad()
  out = net(x)
  loss = F.cross_entropy(out.reshape(-1, n_toks),y.flatten())
  loss.backward()
  opt.step()
  return loss

#%%
epochs = 1000
l = step(x,y)
print(l.item())
for _ in range(10):
  x,y = gen_data(100)
  for _ in range(epochs // 20):
    l = step(x,y)* 0.1 + l * 0.9
  print(l.item())

#%%

print ("******* INFERENCE ********")

def generate(n):
  x = torch.zeros(1, 1).long()
  for i in range(n):
    p = net(x[:,-99:])
    choices = p[0,-1,:]
    choice = torch.multinomial(choices.softmax(-1), 1)
    x = torch.cat([x, choice.reshape(1,1)], dim=1)
    # if (x[:,-2:] == 49).all(): break
  return x

for i in range(5):
  p = generate(max_points*2)
  plt.show()
  display(p[0,1:])

# %%
