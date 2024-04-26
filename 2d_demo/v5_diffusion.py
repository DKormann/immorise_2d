# %%
import numpy as np
import torch 
import matplotlib.pyplot as plt

# %%
class Mesh:
  def __init__(self, edges: torch.Tensor):
    self.edges = edges
    self.edge_lens = torch.norm(edges[:,0] - edges[:,1], dim=1)
  
  def loop(vertices):
    edges = torch.cat([vertices[:-1], vertices[1:]],dim=1).reshape(-1, 2, 2)
    return Mesh(edges)
  
  def plot(self, c='b'):
    for edge in self.edges:
      plt.plot(*edge.T, c)
    
  def pointcloud(self,n=1000,std = 0.01):
    edges_probs = self.edge_lens/self.edge_lens.sum()
    edges = torch.multinomial(edges_probs, n, replacement=True)
    edges = self.edges[edges]
    t = torch.rand(n)
    points = edges[:,0] + t[:,None]*(edges[:,1]-edges[:,0])
    noise = torch.randn_like(points)*std
    return points + noise
  
box = Mesh.loop(torch.tensor([[0, 0], [0, 1], [1, 1],[1,0], [0,0]],dtype=torch.float32))
box.plot()
# %%
def random_shape():
  n = np.random.randint(3, 10)
  corners = []

  for i in range(n):
    angle = 2*np.pi*i/n + np.random.rand()*0.1
    r = np.random.rand() * 0.5 + 0.5
    corners.append(torch.tensor([np.cos(angle)*r, np.sin(angle)*r],dtype=torch.float32))

  verts = []
  corner_dir = np.random.randint(2)
  for i in range(n + 1):
    verts.append(corners[i % n])
    verts.append(torch.tensor([corners[(i+corner_dir) % n][0] , corners[(i + 1 - corner_dir) % n][1]]))

  return Mesh.loop(torch.stack(verts[:-1]))


shape = random_shape()
shape.plot()
v = shape.pointcloud(200)
plt.scatter(*v.T)

# %%

def gen_data(n,t):
  train_shapes = [random_shape() for _ in range(n)]
  x = [s.pointcloud(100) for s in train_shapes]
  y = [s.edges for s in train_shapes]
  x = torch.stack(x)
  maxverts = 18
  y = [torch.cat([y.reshape(-1,4), torch.ones(len(y), 1)], dim=1) for y in y]
  y = torch.stack([torch.cat([y, torch.zeros(maxverts - len(y), 5)]) for y in y])
  s = torch.randn(n, 18, 5) 
  s[:,:,4] = s[:,:,4].sigmoid()
  s = y * t + s * (1-t)    
  s = torch.cat([s, torch.ones(n, 18, 1)*t], dim=2)
  return x, s, y


x,s,y = gen_data(10,.3)
k = 0

edges = y[k].reshape(-1,5)[:,:4].reshape( 18, 2, 2)
pmask = y[k].reshape(-1,5)[:,4].reshape( 18, 1)
for edge in edges: plt.plot(*edge.T, c='black')

sedges = s[k].reshape(-1,6)[:,:4].reshape( 18, 2, 2)
smask = s[k].reshape(-1,6)[:,4].reshape( 18, 1)
for edge,m in zip(sedges,smask): plt.plot(*edge.T, c=plt.colormaps['Blues'](m.item()))

#%%

from torch.nn import TransformerEncoderLayer, Linear, Sequential
from torch.optim import Adam

max_edges = 18
hidden_dim = 1024
nhead = 4
inducing_points = 18

class MAB(torch.nn.Module):
  def __init__(self, hidden_dim, nhead, dropout=0.1):
    super(MAB, self).__init__()
    self.attention = torch.nn.MultiheadAttention(hidden_dim, nhead, batch_first=True, dropout=dropout)
    self.norm = torch.nn.LayerNorm(hidden_dim)
    self.linear = Linear(hidden_dim, hidden_dim)
    
  def forward(self, q, kv): 
    x = self.attention(q, kv, kv)[0]
    x = self.norm(x)
    return self.linear(x).relu()

class Model(torch.nn.Module):
  def __init__(self, hidden_dim, nhead, k):
    super(Model, self).__init__()
    self.pointemb = Linear(2, hidden_dim)
    self.edgeemb = Linear(6, hidden_dim)
    
    self.mab1 = MAB(hidden_dim,nhead,dropout=0.1)
    self.mab2 = MAB(hidden_dim,nhead,dropout=0.1)

    self.adapter = Linear(hidden_dim, 5)


  def forward(self, x, s):
    x = self.pointemb(x)
    p = self.edgeemb(s)    
    p = self.mab1(p,x) + self.mab2(p,p)
    res = self.adapter(p) + s[:,:,:5]
    res[:,:,4] = res[:,:,4].sigmoid()
    return res

  def diffusion(self, x):
    s = torch.randn(x.shape[0], 18, 6) * 0.05
    s[:,:,4] = s[:,:,4].sigmoid()
    s[:,:,5] = 0.

    for i in range(10,0,-1):
      p = torch.cat([self.forward(x, s), torch.ones(x.shape[0], 18, 1)], dim=2)
      s = s * (i-1)/i + p * 1/i

    return s

  def save(self, path): torch.save(self.state_dict(), path)
  def load(self, path): self.load_state_dict(torch.load(path))


train_x, train_s, train_y = gen_data(200,0.1)
model = Model(hidden_dim, nhead, inducing_points)

pred = model.diffusion(train_x[:2])
optimizer = Adam(model.parameters(), lr=0.001)

#%% helper funcs
def loss_fn(p, y):
  mask = y[:,:,4].unsqueeze(2)
  pmask = p[:,:,4].unsqueeze(2)
  dist = (p[:,:,:4] - y[:,:,:4])**2 * mask
  geom_loss = dist.sum() / mask.sum()
  mask_loss = torch.nn.functional.binary_cross_entropy(pmask, mask)
  return geom_loss + mask_loss

def step(t):
  x,s,y = gen_data(200,t)
  p = model(x,s)
  loss = loss_fn(p,y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss.item()


loss_table = [0] * 10
#%%
try:
  
  n = 2000
  for i in range(n):
    t = np.random.randint(0,10)
    l = step(t/10)
    loss_table[t] = loss_table[t]* 0.9 + l*0.1 
    print(f'\repoch {i}/{n} loss: [{" ".join(map(lambda x: str(round(x,3)), loss_table))}] ',end='')
    if i % (2000//10) == 0: print()
except KeyboardInterrupt: pass

#%%

x,s,y = gen_data(1,0.8)
p = model(x,s).detach()

edges = y.reshape(-1,5)[:,:4].reshape( 18, 2, 2)

pedges = p.reshape(-1,5)[:,:4].reshape( 18, 2, 2)
pmask = p.reshape(-1,5)[:,4].reshape( 18, 1)
for edge,m in zip(pedges,pmask): plt.plot(*edge.T, c=plt.colormaps['Blues'](m.item()))

sedges = s.reshape(-1,6)[:,:4].reshape( 18, 2, 2)
smask = s.reshape(-1,6)[:,4].reshape( 18, 1)
for edge,m in zip(sedges,smask): plt.plot(*edge.T, c=plt.colormaps['Greens'](m.item()))
plt.scatter(*x[0].T)

