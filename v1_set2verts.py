# %%
import numpy as np
import torch 
import matplotlib.pyplot as plt


# %%
class Shape:
  def __init__(self, vertices: torch.Tensor):
    self.vertices = vertices
  
  @property
  def loop(self):
    edges = torch.cat([self.vertices[:-1], self.vertices[1:]],dim=1).reshape(-1, 2, 2)
    return edges
  
  def plot(self, c='b'):
    for edge in self.loop:plt.plot(edge[:,0], edge[:,1],c=c)
    
  def pointcloud(self,n=1000,std = 0.01):
    edge_lens = torch.norm(self.vertices[:-1] - self.vertices[1:], dim=1)
    edges_probs = edge_lens/edge_lens.sum()
    edges = torch.multinomial(edges_probs, n, replacement=True)
    edges = self.loop[edges]
    t = torch.rand(n)
    points = edges[:,0] + t[:,None]*(edges[:,1]-edges[:,0])
    noise = torch.randn_like(points)*std
    return points + noise
  
box = Shape(torch.tensor([[0, 0], [0, 1], [1, 1],[1,0], [0,0]],dtype=torch.float32))
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

  return Shape(torch.stack(verts[:-1]))


shape = random_shape()
shape.plot()
v = shape.pointcloud(200)
plt.scatter(*v.T)

# %%
def gen_data(t,e):
  train_shapes = [random_shape() for _ in range(t+e)]
  x = [s.pointcloud(100) for s in train_shapes]
  y = [s.vertices for s in train_shapes]

  x = torch.stack(x)
  maxverts = max([len(x) for x in y])
  y = torch.stack([torch.cat([y, torch.zeros(maxverts - len(y), 2)]) for y in y]).reshape(-1, maxverts*2)
  return x[:t], y[:t], x[t:], y[t:]

# %%
from torch.nn import TransformerEncoderLayer, Linear, Sequential
from torch.optim import Adam

input_dim = 2
output_dim = 19 * 2
hidden_dim = 256
nhead = 2
num_layers = 3

class SetTransformer(torch.nn.Module):
  def __init__(self, input_dim, output_dim, nhead, num_layers):
    super(SetTransformer, self).__init__()
    self.model = Sequential(
      Linear(input_dim, hidden_dim),
      *[TransformerEncoderLayer(hidden_dim, nhead, batch_first=True) for _ in range(num_layers)],
      Linear(hidden_dim, output_dim),
    )
  
  def forward(self, x): return self.model(x).mean(dim=1)

  def save(self, path): torch.save(self.state_dict(), path)

  def load(self, path): self.load_state_dict(torch.load(path))

model = SetTransformer(input_dim, output_dim, nhead, num_layers)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# %%

model.load_state_dict(torch.load('set2vertsmodel.pth'))
# %%

def step(train_x,train_y):
  optimizer.zero_grad()
  p = model(train_x)
  loss = loss_fn(p, train_y)
  loss.backward()
  optimizer.step()
  print(f"\rtrain:{loss.item()} ", end = "")
  return p

def display(p,x,y):
  k = np.random.randint(0, len(p))
  p_shape = Shape(p[k].detach().reshape( 19, 2))
  t_shape = Shape(y[k].reshape( 19, 2))
  t_shape.plot('black')
  p_shape.plot('blue')
  plt.scatter(*x[k].T, c='gray')
#%%

for i in range(50):
  print(f'Run {i}')
  train_x, train_y, test_x, test_y = gen_data(180,20)
  model.train()
  for i in range(10): step(train_x,train_y)
  p = model.eval()(test_x)
  loss = loss_fn(p, test_y)
  print(f"\ntest:{loss.item()}")
  display(p,test_x,test_y)
  plt.show()
  model.save('set2vertsmodel.pth')

#%%
torch.save(model.state_dict(), 'set2vecmodel.pth')
