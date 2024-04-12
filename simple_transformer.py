

#%%
import torch
from torch.nn import functional as F

import matplotlib.pyplot as plt

max_seq_len = 100
dim = 512
hidden_dim = 2048

class Layer(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.W = torch.nn.Linear(dim, hidden_dim)
    self.WK = torch.nn.Linear(hidden_dim, hidden_dim)
    self.WQ = torch.nn.Linear(hidden_dim, hidden_dim)
    self.WV = torch.nn.Linear(hidden_dim, hidden_dim)
    self.W2 = torch.nn.Linear(hidden_dim, dim)

    self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
  def forward(self, x):
    
    attn_mask = torch.where(self.causal_mask[:x.shape[1], :x.shape[1]], 0, -float("inf"))
    residual = x
    
    x = self.W(x)
    x = F.relu(x)
    K = self.WK(x)
    Q = self.WQ(x)
    V = self.WV(x)
    
    attn = Q @ K.transpose(1,2)
    attn = attn / (dim ** 0.5)
    attn = attn + attn_mask

    attn = F.softmax(attn, dim=-1)
    x = attn @ V
    x = self.W2(x) + residual
    return F.layer_norm(x, x.shape[1:])

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding = torch.nn.Embedding(100, dim)
    self.layers = torch.nn.ModuleList([Layer() for _ in range(1)]) 
    self.out = torch.nn.Linear(dim, 100)
  def forward(self, x):
    x = self.embedding(x)
    for layer in self.layers:
      x = layer(x)
    x = self.out(x)
    return x

model = Model()
len([*model.parameters()])


random_x = torch.randint(0, 100, (1, max_seq_len))
model(random_x).shape



    