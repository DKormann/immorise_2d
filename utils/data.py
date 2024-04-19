#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%

max_points = 21

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

  verts = torch.stack(verts[:-1])
  return verts

def pointcloud(verts,n, noise = 0.01):
  edges = torch.cat([verts, verts.roll(-1, 0)], dim=1)
  edge_lens = torch.linalg.norm(edges[:,2:] - edges[:,:2], dim=1)
  rand_edges = edges[torch.multinomial(edge_lens / edge_lens.max(), n, replacement=True)]
  points = rand_edges[:, :2] + torch.rand((n,1)).reshape(-1,1) * (rand_edges[:, 2:] - rand_edges[:, :2])
  return points +torch.randn(n, 2) * noise

n_toks = 100

def tokenize(x): return ((x + 1) / 2 * (n_toks-1)).clamp(1, n_toks-1).long()

max_points = 21
def gen_data(n, randorder = False):
  shapes = [random_shape() for _ in range(n)]
  pts = tokenize(torch.stack([(pointcloud(s, 100)) for s in shapes]))
  if randorder: shapes = [s[torch.randperm(len(s))] for s in shapes]
  y = torch.stack([torch.cat([tokenize(s), torch.zeros(max_points - len(s), 2)]) for s in shapes]).view(n, -1).long()
  pad = torch.zeros(n, 1).long()
  x = torch.cat([pad, y], dim=1)[:, :-1]
  return x, y, pts


def display(shape):
  shape = shape.where(shape != 0, torch.tensor(np.nan)).cpu()
  plt.plot(shape[::2], shape[1::2])


if __name__ == '__main__':
    x, y, pts = gen_data(1)
    display(y[0])
    plt.scatter(*(pts.T.cpu()+1)*50)
    