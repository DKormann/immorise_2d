

#%%
import torch
from torch.nn import functional as F


max_seq_len = 100

dim = 512
hidden_dim = 2048

class Layer(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.W = torch.nn.Linear(dim, hidden_dim)
    self.WK = torch.nn.Linear(hidden_dim, hidden_dim)
    self.QK = torch.nn.Linear(hidden_dim, hidden_dim)
    self.WV = torch.nn.Linear(hidden_dim, hidden_dim)

    self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
  def forward(self, x):
    
    attn_mask = torch.where(self.causal_mask[:x.shape[1], :x.shape[1]], 0, -float("inf"))
    residual = x
    x = self.W(x)
    x = F.relu(x)
    x = x + residual
    residual = x
    x = F.relu(x)
    K = self.WK(x)
    Q = self.QK(x)
    V = self.WV(x)
    
    attn = Q @ K.T / (dim ** 0.5)
    attn = attn + attn_mask
    attn = F.softmax(attn, dim=-1)
    x = attn @ V
    x = x + residual
    return x

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding = torch.nn.Embedding(100, dim)
    self.layers = torch.nn.ModuleList([Layer() for _ in range(8)]) 
    self.out = torch.nn.Linear(dim, 100)
  def forward(self, x):
    x = self.embedding(x)
    for layer in self.layers:
      x = layer(x)
    x = self.out(x)
    return x

model = Model()
len([*model.parameters()])
    
    
    