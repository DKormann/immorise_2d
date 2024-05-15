#%%
import torch 
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

torch.set_default_device('cuda')
device = torch.device('cuda')

ds = MNIST('mnist', download=True).data.float().cuda()/256
ds = ds.view(-1, 28*28)
points = torch.multinomial(ds, 100, replacement=True).float() 
points_x = points % 28 + torch.randn_like(points) * 0.1
points_y = 28 - points // 28 + torch.randn_like(points) * 0.1
points = torch.stack([points_x, points_y], dim=-1)
labels = MNIST('mnist', download=True).targets.cuda()
# %%

dim = 1028

class TransLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.expand = nn.Linear(dim, dim*4)
    self.out = nn.Linear(dim*4, dim)
  
  def forward(self, x):
    residual = x
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)
    w = F.softmax(q @ k.transpose(-2,-1) / (dim ** 0.5), dim=-1)
    x = self.out(F.gelu(self.expand(w @ v)))
    return x + residual

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Linear(2, dim)
    self.layers = nn.Sequential(*[TransLayer() for _ in range(6)])
    self.out = nn.Linear(dim, 10)
  def forward(self, x):
    x = self.emb(x)
    x = self.layers(x)
    return self.out(x).mean(1)
model = Model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)



# %%
epochs = 10
batch_size = 64
n_batches = len(points) // batch_size

for e in range(epochs):
  for i in range(n_batches):
    x = points[i*batch_size:(i+1)*batch_size]
    y = labels[i*batch_size:(i+1)*batch_size]
    p = model(x)
    loss = F.cross_entropy(p, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    print('\r', loss.item(), end='')
  # if e % (epochs//10) == 0:
  print()
