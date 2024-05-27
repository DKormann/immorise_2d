#%%
import os
if not os.path.exists('floor_decoder'):
  os.chdir('..')

from floor_decoder.b_image import idata, shapes

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt


torch.set_default_device('cuda')
idata = idata.cuda().reshape(idata.shape[0], 8*8, np.prod(idata.shape[-2:]))



# %%

maxlen = max([len(shape) for shape in shapes])
maxval = max([shape.max() for shape in shapes])
ntoks = 100

def discretize(shape:np.ndarray):
  shape = shape / maxval * (ntoks-2) + 1
  return torch.tensor(shape, dtype=torch.long).clamp(1, ntoks-1)

sdata = [discretize(shape) for shape in shapes]
for s in sdata: s[:,1::2] = ntoks - s[:,1::2]

sdata = torch.stack([F.pad(shape, (0, 0, 0, maxlen - len(shape))) for shape in sdata]).reshape(-1, maxlen*4)

#%%

test_idata, idata = idata[:5], idata[5:]
test_sdata, sdata = sdata[:5], sdata[5:]


#%%
def plot_image(img):
  fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
  axes = axes.flatten()

  img = img.reshape(64,252, 366)

  for i, ax in enumerate(axes):
      ax.imshow(img[i].cpu(), cmap='gray')
      ax.axis('off')  # Hide the axis

  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  plt.show()

# for i in range(29):show_image(idata[i])


#%%

def plot_shape(shape):
  shape = shape.view(-1,2,2).cpu()
  idx = (shape==0).nonzero()
  if len(idx) >0:shape = shape[:idx[0,0]]
  plt.axis('equal')
  for i,e in enumerate(shape):plt.plot(e[:,0],e[:,1], color=plt.cm.viridis(i/len(shape)))
  plt.show()



#%%

k = 4
plot_image(idata[k])
plot_shape(sdata[k])
#%%

# max_points = maxlen*2

max_seq_len = maxlen*4 + 8**2

model_scale = 1.5
to_nearest_64 = lambda x: round(x/64) * 64
residual_depth = to_nearest_64(384 * np.log2(1.+model_scale))
num_blocks = round(8 * np.log2(1.+model_scale))
qk_dim_div = 8
expand_factor = 2
def init_weights(*size): return nn.Parameter(nn.init.xavier_uniform_(torch.empty(*size)))


class AttentionLayer(nn.Module):
  def __init__(self, num_dim=residual_depth, max_seq_len= max_seq_len, **kwargs):
    super().__init__()
    self.dim        = num_dim
    self.qk_dim     = self.dim // qk_dim_div
    self.expand_dim = num_dim * expand_factor
    self.expand     = init_weights(self.dim,2*self.qk_dim+2*self.expand_dim)
    self.project    = init_weights(self.expand_dim, self.dim)
    with torch.no_grad():
      self.position_bias_base = torch.arange(0, max_seq_len).unsqueeze(0).float() - torch.arange(0, max_seq_len).unsqueeze(1)
      self.negative_infinity_matrix_base = torch.empty_like(self.position_bias_base).fill_(-float("inf"))
      self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
    self.position_bias_mult = nn.Parameter(torch.tensor(4.))
  def get_mask(self, n): return torch.where(self.causal_mask[:n, :n], (self.position_bias_base[:n,:n] + self.position_bias_mult).sigmoid(), self.negative_infinity_matrix_base[:n, :n])
  def forward(self, x):
    nx = F.layer_norm(x, [self.dim])
    q, k, linear, pre_gelu = (nx @ self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), -1)
    local, v = (linear * F.gelu(pre_gelu)).split((self.expand_dim-self.dim, self.dim), -1)
    scores = torch.softmax((q @ k.transpose(-2, -1) / q.size(-1)**.5) + self.get_mask(x.shape[1]), -1)
    return x + torch.cat([local, scores @ v], -1) @ self.project

layer  = AttentionLayer()
assert layer(torch.randn(1, max_seq_len, residual_depth)).shape == torch.Size([1, max_seq_len, residual_depth])

class Model(nn.Module):
  def __init__(self, max_seq_len=max_seq_len):
    super().__init__()
    self.freqs = torch.exp(torch.arange(0, residual_depth, 2).float() * -(9. / residual_depth))
    self.linpatchenc = init_weights(idata.shape[-1], residual_depth)
    self.pospatchenc = init_weights(8*8, residual_depth)

    self.blocks = nn.Sequential(*[AttentionLayer(residual_depth,max_seq_len=max_seq_len) for _ in range(num_blocks)])
    self.out = nn.Linear(residual_depth, 100, bias=False)
    self.norm = nn.LayerNorm(residual_depth)
  
  def forward(self, q:torch.Tensor, patches:torch.Tensor):
    patches = patches @ self.linpatchenc
    patches = patches + self.pospatchenc
    q = self.freqs * q.unsqueeze(-1)
    q = torch.cat([torch.sin(q), torch.cos(q)], -1)
    q = torch.cat([patches, q], 1)
    r = self.blocks(q)
    r = self.norm(r)
    return self.out(r)

net = Model()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
assert net(torch.zeros(1, maxlen*4), idata[0:1]).shape == torch.Size([1, max_seq_len, 100])


#%%
opt = torch.optim.Adam(net.parameters(), lr=1e-5)

epochs = 2000
batch_size = 4
n_batches = len(sdata) // batch_size + 1
if __name__ == '__main__':
  # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=n_batches, epochs=epochs, pct_start=0.1)
  try:
    for e in range(epochs):
      for i in range(0,len(sdata),batch_size):
        y = sdata[i:i+batch_size]
        patches = idata[i:i+batch_size]

        x = F.pad(y, (1, 0, 0, 0))[:,:-1]
        opt.zero_grad()
        p = net(x.cuda(), patches.cuda())[:,8*8:]
        loss = F.cross_entropy(p.reshape(-1, 100),y.flatten().cuda())
        loss.backward()
        opt.step()
        print('\r', loss.item(),end='')
        if (e % (10) == 0) and (i == 0):
          with torch.no_grad():
            test_y = F.pad(test_sdata, (1, 0, 0, 0))[:,:-1]
            test_p = net(test_y.cuda(), test_idata.cuda())[:,8*8:]
            test_loss = F.cross_entropy(test_p.reshape(-1, 100),test_y.flatten().cuda())
            print(f' e: {e} val: {test_loss.item()}')
        # scheduler.step()
  except KeyboardInterrupt: pass

# %%
opt = torch.optim.Adam(net.parameters(), lr=1e-5)

if __name__ == '__main__':
  print(" ********* Inference *********")
  for i in range(1):
    x = torch.zeros(1,1).long()
    for i in range(maxlen*4):
      k = 2
      out = net(x,test_idata[k:k+1])[0,-1:]
      out = torch.argmax(out).view(1,1)
      x = torch.cat([x, out[0].view(1,1)],1)

    d = x[0,1:]
    # d = d.where(d!=0, torch.tensor(torch.nan))
    d = d[:len(d)//4*4].view(-1,2,2).cpu()
    plot_shape(d)
    plot_image(test_idata[k])

#%%

plt.plot(d.reshape(-1))
#%%

# get index of d where d is first nan
torch.isnan(d).nonzero()[0,0]
