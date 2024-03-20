# %%
import numpy as np
import torch 
import matplotlib.pyplot as plt


# %%
class Shape:
  def __init__(self, edges: torch.Tensor):
    self.edges = edges
    self.edge_lens = torch.norm(edges[:,0] - edges[:,1], dim=1)
  
  def loop(vertices):
    edges = torch.cat([vertices[:-1], vertices[1:]],dim=1).reshape(-1, 2, 2)
    return Shape(edges)
  
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
  
box = Shape.loop(torch.tensor([[0, 0], [0, 1], [1, 1],[1,0], [0,0]],dtype=torch.float32))
box.plot()
# %%
def random_shape():
  n = np.random.dint(3, 10)
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

  return Shape.loop(torch.stack(verts[:-1]))


shape = random_shape()
shape.plot()
v = shape.pointcloud(200)
plt.scatter(*v.T)

# %%
train_shapes = [random_shape() for _ in range(200)]
train_x = [s.pointcloud(100) for s in train_shapes]
train_y = [s.edges for s in train_shapes]

train_x = torch.stack(train_x)
maxverts = max([len(x) for x in train_y])

train_y = torch.stack([torch.cat([y, torch.zeros(maxverts - len(y), 2, 2)]) for y in train_y]).reshape(-1, maxverts*4)
# %%
from torch.nn import TransformerEncoderLayer, Linear, Sequential
from torch.optim import Adam

input_dim = 2
output_dim = 18 * 4
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

model = SetTransformer(input_dim, output_dim, nhead, num_layers)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# %%

def step(train_x,train_y):
  optimizer.zero_grad()
  p = model(train_x)
  loss = loss_fn(p, train_y)
  loss.backward()
  optimizer.step()
  print(loss.item())
  return p

def display(p):
  k = np.random.randint(0, len(train_x))
  p_shape = Shape(p[k].detach().reshape( 18, 2, 2))
  t_shape = train_shapes[k]
  t_shape.plot('black')
  p_shape.plot('blue')
  plt.scatter(*train_x[k].T, c='gray')
#%%

for _ in range(20):
  p = None
  for i in range(10):p=step(train_x,train_y)
  display(p)
  plt.show()

#%%
torch.save(model.state_dict(), 'set2vecmodel.pth')
