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

# %%

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(28*28, 1024),
      nn.GELU(),
      nn.Linear(1024, 1024),
      nn.GELU(),
      nn.Linear(1024, 28*28),
      nn.Sigmoid()
      )
  def forward(self, x):
    input_shape = x.shape
    if x.shape[-1] != 28*28:x = x.view(-1, 28*28)
    for layert in self.layers:
      x = layert(x)
    return x.view(*input_shape)

net = Model().cuda().train()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# %%
bi = 0
alpha = 0.5
alphas = [0.98**t for t in range(1,10)]+[0.]

#%%

for i in range(1000):
  bi = (bi+1)%(len(train)//100)
  x0 = train[bi*100:(bi+1)*100]
  t = torch.randint(0,10,()).item()
  
  x_t = alphas[t]*x0 + math.sqrt(1-alphas[t])*torch.randn_like(x0)

  p = net(x_t)
  loss = F.mse_loss(p, x0)
  opt.zero_grad()
  loss.backward()
  opt.step()
print(t, loss.item())
plt.imshow(x_t[0].detach().cpu())
plt.show()
plt.imshow(p[0].detach().cpu())
plt.show()


#%%

print(" ********* Inference *********")
for i in range(1):
  x = torch.randn(1,28*28).cuda()
  for t in range(9,1,-1):


    sigma = math.sqrt(((1-alpha)*(1-math.sqrt(alphas[t-1]))) / (1-alphas[t]))

    x = \
      (1-alphas[t-1]) * math.sqrt(alphas[t])/(1-alphas[t]) * x + \
      (1-alphas[t-1]) * math.sqrt(alphas[t-1])/(1-alphas[t]) * net(x) + \
      sigma*torch.randn_like(x)
  
    plt.imshow(x[0].detach().cpu().view(28,28))
    plt.show()
  
  x = net(x)
  plt.imshow(x[0].detach().cpu().view(28,28))
  plt.show()
      
