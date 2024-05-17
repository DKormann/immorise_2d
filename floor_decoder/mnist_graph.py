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
points_y = 28 - points // 28 #+ torch.randn_like(points) * 0.1
points = torch.stack([points_x, points_y], dim=-1)
labels = MNIST('mnist', download=True).targets.cuda()

# %%

dim = 1024

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
    self.q = nn.Linear(dim, dim//2)
    self.k = nn.Linear(dim, dim//2)
    self.v = nn.Linear(dim, dim)
    self.expand = nn.Linear(dim, dim*4)
    self.out = nn.Linear(dim*4, dim)
    self.norm = nn.LayerNorm(dim)

  
  def forward(self, x, plot = False):
    residual = x
    nheads = 1
    q = self.q(x)#.view(*x.shape[:-1], nheads, -1)
    k = self.k(x)#.view(*x.shape[:-1], nheads, -1)
    v = self.v(x)#.view(*x.shape[:-1], nheads, -1)
    
    scores = torch.einsum('bqd,bkd->bkq', q, k) / math.sqrt(dim//4)
    scores = scores.softmax(1)
    if plot: 
      plt.imshow(scores[0].detach().cpu())
      plt.show()
      print(scores[0].cpu())

    x = torch.einsum('bkq,bkd->bqd', scores, v).reshape(*v.shape[:2],-1)
    x = self.norm(x+residual)
    x = self.out(F.gelu(self.expand(x)))+x
    return x

layer = TransLayer()
assert layer(enc(points[:2])).shape == torch.Size([2, 100, dim])

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = PointEnc()
    self.layers = nn.Sequential(*[TransLayer() for _ in range(8)])
    self.out = nn.Linear(dim, 10)
    self.clstoken = nn.Parameter(torch.randn(1, 1, dim))
  def forward(self, x, plot = False):
    x = self.emb(x)
    x = torch.cat([self.clstoken.expand(x.shape[0], -1, -1), x], 1)
    # x = self.layers(x)
    for layer in self.layers:
      x = layer(x,plot)
    return self.out(x)[:,0]
model = Model()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

assert model(points[:2],True).shape == torch.Size([2, 10])

# %%
epochs = 1
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
      print(f'\r{i}/{n_batches}', loss.item(), end='')
    print()
  except KeyboardInterrupt: break

# %%


k = 5
plt.xlim(0, 28)
plt.ylim(0, 28)
plt.scatter(*points[k].T.cpu())
plt.show()
plt.plot(model(points[k:k+1]).detach().cpu()[0])
labels[k]
# %%
a = torch.rand(10,10)
a = a.softmax(-1)
a.sum(0)