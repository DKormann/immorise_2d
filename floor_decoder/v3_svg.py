#%%
import os
if not os.path.exists('floor_decoder'):
  os.chdir('..')

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from data_loader.read_svgs import imsize, n_toks, max_edges, get_test_batch, get_train_batch

# %%
if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"
   
torch.set_default_device(device)

# %%
images, edg = get_train_batch()
images = images.to(device)
edg = edg.to(device)

splits = 10

def split_image(img):
  return img.reshape(-1, splits, 1000//splits, splits, 1000//splits).permute(0,1,3,2,4).reshape(-1, splits**2, 1000//splits, 1000//splits)

img = split_image(images[:10])
def plot_image(img):
  fig, axes = plt.subplots(nrows=splits, ncols=splits, figsize=(10,10))
  axes = axes.flatten()
  for i, ax in enumerate(axes):
    ax.imshow(img[i].cpu(), cmap='gray',vmin=0, vmax=1)

    ax.axis('off')
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  plt.show()

plot_image(img[0])

#%%


def plot_shape(shape):
  shape = shape.view(-1,2,2).cpu()
  idx = (shape==0).nonzero()
  if len(idx) >0:shape = shape[:idx[0,0]]
  plt.axis('equal')
  for i,e in enumerate(shape):plt.plot(e[:,0],100-e[:,1], color=plt.cm.viridis(i/len(shape)))
  plt.show()


plot_shape(edg[0])

#%%

max_seq_len = max_edges*4 + splits**2

model_scale = 1.5
to_nearest_64 = lambda x: round(x/64) * 64
residual_depth = to_nearest_64(384 * np.log2(1.+model_scale))
num_blocks = round(8 * np.log2(1.+model_scale))
qk_dim_div = 8
expand_factor = 2
def init_weights(*size): return nn.Parameter(nn.init.xavier_uniform_(torch.empty(*size)))


class AttentionLayer(nn.Module):
  def __init__(self, num_dim=residual_depth, max_seq_len= max_seq_len, dropout=0.0, **kwargs):
    super().__init__()
    self.dim        = num_dim
    self.qk_dim     = self.dim // qk_dim_div
    self.expand_dim = num_dim * expand_factor
    self.expand     = init_weights(self.dim,2*self.qk_dim+2*self.expand_dim)
    self.project    = init_weights(self.expand_dim, self.dim)
    with torch.no_grad():
      self.position_bias_base = torch.arange(0, max_seq_len).unsqueeze(0).float() - torch.arange(0, max_seq_len).unsqueeze(1)
      self.position_bias_base[:,:100] = -10
      self.negative_infinity_matrix_base = torch.empty_like(self.position_bias_base).fill_(-float("inf"))
      self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
      self.causal_mask[:,:splits**2] = True
    self.position_bias_mult = nn.Parameter(torch.tensor(4.))
    self.dropout = nn.Dropout(dropout)
  def get_mask(self, n): return torch.where(self.causal_mask[:n, :n], (self.position_bias_base[:n,:n] + self.position_bias_mult).sigmoid(), self.negative_infinity_matrix_base[:n, :n])
  def forward(self, x):
    nx = F.layer_norm(x, [self.dim])
    nx = self.dropout(nx)
    q, k, linear, pre_gelu = (nx @ self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), -1)
    local, v = self.dropout(linear * F.gelu(pre_gelu)).split((self.expand_dim-self.dim, self.dim), -1)
    scores = torch.softmax((q @ k.transpose(-2, -1) / q.size(-1)**.5) + self.get_mask(x.shape[1]), -1)
    return x + torch.cat([local, scores @ v], -1) @ self.project

layer  = AttentionLayer()
plt.imshow(layer.get_mask(200).cpu().detach().numpy())
assert layer(torch.randn(1, max_seq_len, residual_depth)).shape == torch.Size([1, max_seq_len, residual_depth])

#%%

class Model(nn.Module):
  def __init__(self, max_seq_len=max_seq_len):
    super().__init__()
    self.freqs = torch.exp(torch.arange(0, residual_depth, 2).float() * -(9. / residual_depth))
    # self.linpatchenc = init_weights(100**2, residual_depth)
    self.encoder = nn.ModuleList([
      nn.Conv2d(1, 64, 3, 2, 1),
      nn.Conv2d(64, 128, 3, 2, 1),
      nn.Conv2d(128, 512, 3, 2, 1),
    ])
    for p in self.encoder.parameters():
      if len(p.shape) > 1:
        nn.init.xavier_uniform_(p)
    self.pospatchenc = init_weights(100, residual_depth)

    self.blocks = nn.Sequential(*[AttentionLayer(residual_depth,max_seq_len=max_seq_len) for _ in range(num_blocks)])
    self.out = nn.Linear(residual_depth, 100, bias=False)
    self.norm = nn.LayerNorm(residual_depth)
  
  def forward(self, q:torch.Tensor, patches:torch.Tensor):
    patches = patches.reshape(-1,1, 100, 100).float()
    for enc in self.encoder:
      patches = enc(patches)
      patches = F.relu(patches)
      patches = F.max_pool2d(patches, 2, 2)
    patches = patches.reshape(-1, splits**2, 512)
    patches = patches + self.pospatchenc
    q = self.freqs * q.unsqueeze(-1)
    q = torch.cat([torch.sin(q), torch.cos(q)], -1)
    q = torch.cat([patches, q], 1)
    r = self.blocks(q)
    r = self.norm(r)
    return self.out(r)

net = Model()
net.to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)

assert net(torch.zeros(2, max_edges*4), img[:2]).shape == torch.Size([2, max_seq_len, 100])

#%%
opt = torch.optim.Adam(net.parameters(), lr=1e-5)

epochs = 1000
batch_size = 8
n_batches = len(images) // batch_size + 1
if __name__ == '__main__':
  scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=n_batches, epochs=epochs, pct_start=0.1)
  if not "L" in globals() : L = 0
  try:
    for e in range(epochs):
      img, edg = get_train_batch()
      for i in range(0,len(images),batch_size):
        patches = split_image(img[i:i+batch_size])
        y = edg[i:i+batch_size]
        x = F.pad(y, ( 1, 0, 0, 0))[:,:-1]
        opt.zero_grad()
        p = net(x.cuda(), patches.cuda())[:,100:]
        loss = F.cross_entropy(p.reshape(-1, 100),y.flatten().long())
        loss.backward()
        L = L * 0.99 + loss.item() * 0.01
        print(f'\r e: {e} i: {i} loss: {L}', end='')
        opt.step()
        scheduler.step()
      if (e % (10) == 0):
        with torch.no_grad():
          net.train(False)
          test_img, test_y = get_test_batch()
          test_patches = split_image(test_img)
          test_p = net(F.pad(test_y, ( 1, 0, 0, 0))[:,:-1], test_patches.cuda())[:,100:]
          test_loss = F.cross_entropy(test_p.reshape(-1, 100),test_y.flatten().long())
          print(f' test_loss: {test_loss.item()}')
          net.train(True)
  except KeyboardInterrupt: pass

# %%
print(" ********* Inference *********")
img, edg = get_train_batch()
img = split_image(img[:1])
x = torch.zeros(1,1).long()
net.train(False)
for i in range(max_edges*4):
  out = net(x.cuda(), img.cuda())[0,-1:]
  out = torch.argmax(out).view(1,1)
  x = torch.cat([x, out[0].view(1,1)],1)

plot_image(img[0])
plot_shape(x[0,1:])
# %%
