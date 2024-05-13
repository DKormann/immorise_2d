#%% 
import torch 
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math

import torchvision.datasets.mnist as mnist
ds = mnist.MNIST('mnist', train=True, download=True).data
ds = ds.float().cuda()/256

train, test = ds.data[:50000], ds.data[50000:]
torch.set_default_device('cuda')

#%%

class Autoencoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.enc = nn.Sequential(
      nn.Linear(28*28, 128),
      nn.GELU(),
      nn.Linear(128, 128),
      nn.GELU(),
      nn.Linear(128, 10),
      nn.GELU(),
    )
    self.dec = nn.Sequential(
      nn.Linear(10, 128),
      nn.GELU(),
      nn.Linear(128, 128),
      nn.GELU(),
      nn.Linear(128, 28*28),
      nn.Sigmoid()
    )
  def forward(self, x):
    input_shape = x.shape
    if x.shape[-1] != 28*28:x = x.view(-1, 28*28)
    if x.dtype != torch.float32: x = x.float()
    return self.dec(self.enc(x)).view(*input_shape)
  

net = Autoencoder().cuda().train()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
assert net(train[:1].float()).shape == train[:1].shape
# net.enc(train[:1].float()).shape
#%%

epochs = 10
for e in range(epochs):
  for batch in range(100):
    opt.zero_grad()
    data = train[batch*100:(batch+1)*100].float()

    p = net(data)
    loss = F.mse_loss(p, data)
    
    loss.backward()
    opt.step()
  print(loss.item())

#%%

k = torch.randint(0, 100, (1,)).item()
p = net(test[k].float())
plt.imshow(test[k].cpu())
plt.show()
plt.imshow(p.detach().cpu())
