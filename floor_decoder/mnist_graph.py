#%%
import torch
from torchvision.datasets import MNIST

torch.set_default_device('cuda')
labels = MNIST('mnist', download=True).targets.cuda()
points = torch.multinomial((MNIST('mnist').data.float().cuda()/256).view(-1,28*28), 100, replacement=True).float() 
points = torch.stack([points % 28, 28 - points // 28], dim=-1)+torch.rand(*points.shape,2)*0.1

D, layers, B = 1024, 4, 100
init = lambda *s: torch.nn.init.xavier_uniform_(torch.empty(*s,requires_grad=True))
weights = [[init(D, D*3) , init(D, D*4), init(D*4,(10 if i == layers-1  else D))] for i in range(layers)]

def forward(x):
  x = x.unsqueeze(-1) * torch.exp(torch.arange(0, D, 4).float() * -(7. / D))
  x = torch.cat([torch.sin(x), torch.cos(x)], -1).view(*x.shape[:-2],-1)
  for qkv, l1, out in weights:
    q,k,v = (x@qkv).chunk(3, dim=-1)
    x = (((x + ((q @ k.transpose(-2, -1)) / (q.shape[-1])**.5).softmax(-1) @ v) @l1).relu()) @ out
  return x.mean(1)

opt = torch.optim.Adam(sum(weights, []), lr=1e-6)
for i in range(0, len(points), B):
  x,y = points[i:i+B], labels[i:i+B]
  loss = torch.nn.functional.cross_entropy(forward(x), y)
  opt.zero_grad()
  loss.backward()
  opt.step()
  print(f'\r{i}/{len(points)}', loss.item(), end='')

#%%
# import matplotlib.pyplot as plt
# k = 0
# plt.xlim(0, 28)
# plt.ylim(0, 28)
# plt.scatter(*points[k].T.cpu())
# plt.show()
# plt.plot(forward(points[k:k+1]).detach().cpu()[0])
# labels[k]

# %%
