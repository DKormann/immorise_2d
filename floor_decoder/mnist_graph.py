#%%
import torch 
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import math

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

class PointEnc(nn.Module):
  def __init__(self): 
    super().__init__()
    self.freqs = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    self.w = nn.Parameter(torch.randn(2*dim, dim)*dim**-.5)

  def __call__(self, q:torch.Tensor): 
    q = q.unsqueeze(-1) * self.freqs
    q = torch.cat([torch.sin(q), torch.cos(q)], -1).view(*q.shape[:-2], 2*dim)
    return q @ self.w

enc = PointEnc()
assert enc(points[:2]).shape == torch.Size([2, 100, dim])

class TransLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.expand = nn.Linear(dim, dim*4)
    self.out = nn.Linear(dim*4, dim)
    self.norm = nn.LayerNorm(dim)
  
  def forward(self, x):
    residual = x
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)
    w = F.softmax(q @ k.transpose(-2,-1) / (dim ** 0.5), dim=-1)
    x = self.norm(w@v)
    x = self.out(F.gelu(self.expand(x)))
    x = self.norm(x)
    return x + residual


layer = TransLayer()
assert layer(enc(points[:2])).shape == torch.Size([2, 100, dim])

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    # self.emb = nn.Linear(2, dim)
    self.emb = PointEnc()
    self.layers = nn.Sequential(*[TransLayer() for _ in range(6)])
    self.out = nn.Linear(dim, 10)
    self.clstoken = nn.Parameter(torch.randn(1, 1, dim))
  def forward(self, x):
    x = self.emb(x)
    x = torch.cat([self.clstoken.expand(x.shape[0], -1, -1), x], 1)
    x = self.layers(x)
    return self.out(x)[:,0]
model = Model()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

assert model(points[:2]).shape == torch.Size([2, 10])

# %%
epochs = 10
batch_size = 100
n_batches = len(points) // batch_size
opt.param_groups[0]['lr'] = 1e-3 
for e in range(epochs):
  try:
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
  except KeyboardInterrupt: break

# %%
k = 4
plt.scatter(*points[k].T.cpu())
plt.show()
plt.plot(model(points[k:k+1]).detach().cpu()[0])
# %%
