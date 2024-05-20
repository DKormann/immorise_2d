#%%
import torch 
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

torch.set_default_device('cuda')
ds = MNIST('mnist', download=True)
labels = ds.targets.cuda()
points = torch.multinomial((ds.data.float().cuda()/256).view(-1,28*28), 100, replacement=True).float() 
points = torch.stack([points % 28, 28 - points // 28], dim=-1)+torch.rand(*points.shape,2)*0.1

# %%
dim, layers, batch_size = 1024, 4, 100
freqs = torch.exp(torch.arange(0, dim, 4).float() * -(7. / dim))

init = lambda *s: nn.init.xavier_uniform_(torch.empty(*s,requires_grad=True))
weights = [[init(dim, dim*3) , init(dim, dim*4), init(dim*4,(10 if i == layers-1  else dim))] for i in range(layers)]

def forward(x):
  x = x.unsqueeze(-1) * freqs
  x = torch.cat([torch.sin(x), torch.cos(x)], -1).view(*x.shape[:-2],-1)
  for qkv, l1, out in weights:
    q,k,v = (x@qkv).chunk(3, dim=-1)
    x = (((x + F.softmax((q @ k.transpose(-2, -1)) / (q.shape[-1])**.5,-1) @ v) @l1).relu()) @ out
  return x.mean(1)

opt = torch.optim.Adam(sum(weights, []), lr=1e-6)
for i in range(0, len(points), batch_size):
  x = points[i:i+batch_size]
  y = labels[i:i+batch_size]
  loss = F.cross_entropy(forward(x), y)
  opt.zero_grad()
  loss.backward()
  opt.step()
  print(f'\r{i}/{len(points)}', loss.item(), end='')

#%%

# k = 0
# plt.xlim(0, 28)
# plt.ylim(0, 28)
# plt.scatter(*points[k].T.cpu())
# plt.show()
# plt.plot(forward(points[k:k+1]).detach().cpu()[0])
# labels[k]

# %%
