# %%
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from itertools import cycle

torch.set_default_device('cuda')
device = torch.device('cuda')

# if import utils fails change directory to parent folder 
from utils.data import gen_data, display, max_points
from utils.latent_attention import LatentAttentionBlock
#%%
def genfn(n):return gen_data(n, False,True)
batch_size = 100
n_batches = 200
train_data = cycle([genfn(batch_size) for _ in range(n_batches)])
test_data = [genfn(100)]

#%%
x,y,_ = gen_data(1,False,True,False)

model_scale = 1.
to_nearest_64 = lambda x: round(x/64) * 64
residual_depth = to_nearest_64(384 * np.log2(1.+model_scale))
num_blocks = round(8 * np.log2(1.+model_scale))

#%%


position_bias_base = torch.zeros(max_points*2, max_points*2)
inner_boost = torch.eye(max_points*2,max_points*2+1)[:,1:]
position_bias_base[::2] += inner_boost[::2]
# position_bias_base[1::2] += inner_boost[::2]
# position_bias_base += inner_boost

plt.imshow(position_bias_base.cpu())

#%%
class Layer(LatentAttentionBlock):
  def __init__(self, num_dim=residual_depth):
    super().__init__(residual_depth)
    self.position_bias_base = position_bias_base
#%%

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(100, residual_depth, scale_grad_by_freq=True)
    self.blocks = nn.ModuleList([Layer(residual_depth) for _ in range(num_blocks)])
    self.norm = nn.LayerNorm(residual_depth)
    self.out = nn.Linear(residual_depth, 100, bias=False)
  
  def forward(self, q:torch.Tensor):
    q = self.emb(q)
    q = self.norm(q)
    for block in self.blocks:
      q = block(q)
    q = self.norm(q)
    q = self.out(q)
    return q
# %%
net = Model().cuda().train()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=n_batches, epochs=1000, pct_start=0.1)
sum(np.prod(p.shape) for p in net.parameters())
#%%

epochs = 2000

for i in range(epochs):
  x,y,_ = next(train_data)
  opt.zero_grad()
  out = net(x.cuda())
  loss = F.cross_entropy(out.reshape(-1, 100),y.flatten().cuda())
  loss.backward()
  opt.step()
  print('\r', loss.item(),end='')
  if (i+1) %(epochs//10) == 0:
    print('test loss:', F.cross_entropy(net(test_data[0][0]).view(-1,100), test_data[0][1].view(-1)).item())
  scheduler.step()
# %%

print(" ********* Inference *********")
for i in range(10):
  x = torch.zeros(1,1).long()
  for i in range(max_points*2):
    out = net(x)[0,-1:]
    out = torch.multinomial(F.softmax(out, dim=-1), 1)
    x = torch.cat([x, out[0].view(1,1)],1)

  d = x[0,1:-1]
  d = d.where(d != 0, torch.tensor(np.nan)).cpu()
  if d.shape[0] % 2 == 1:
    d = d[:-1]
  d = d.view(-1,2).T
  plt.plot(d[0],d[1])
  plt.show()

# %%

emb = net.emb.weight.detach().cpu()
plt.imshow(emb)
plt.show()

k=60
sim = emb[k]
sim = torch.matmul(emb, sim)
