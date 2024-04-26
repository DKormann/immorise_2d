#%%
import os
import urllib
import re
import pathlib
# datapath = "../dataset"
datapath = pathlib.Path("./dataset")

import numpy as np
import torch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#%%

datasets = [f for f in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, f))]

train_dataset = datasets[:-2]
val_dataset = datasets[-2:-1]
test_dataset = datasets[-1:]

def extract_floors(dataset):
  floors = []  
  for ds in dataset:
    files = os.listdir(os.path.join(datapath, ds))

    for file in files:
      if re.match(r'.*edge_vertices_[0-9]+\.txt', file):
        path  = datapath/ds/file
        try: floors.append(np.loadtxt(path))
        except: print(f"Error loading floor at {path}")
  return floors

def extract_pts(dataset):
  clouds = []
  for ds in dataset:
    files = os.listdir(os.path.join(datapath, ds))

    for file in files:
      if re.match(r'.*_floor_ptcl_[0-9]+\.txt', file):
        path  = datapath/ds/file
        try: clouds.append(np.loadtxt(path))
        except: print(f"Error loading cloud at {path}")
  return clouds

def get_floors(): return extract_floors(train_dataset), extract_floors(val_dataset), extract_floors(test_dataset)
train_floors, val_floors, test_floors = get_floors()
#%%


def display_floor(floor, c = 'b'):
  for edge in floor.reshape(-1,4):
    plt.plot(edge[::2], edge[1::2], c)


ntoks = 100


tok_floors = []
for floor in train_floors:
  floor = torch.tensor(floor).view(-1,2,2)
  #normalize
  fmin = floor.min(0)[0].min(0)[0]
  fmax = floor.max(0)[0].max(0)[0]
  maxsize = (fmax - fmin).max()
  floor = (floor - fmin) / maxsize

  #quantize
  floor = (floor * (ntoks-1)).round().long()
  floor = floor.clamp(1,ntoks-1)
  
  

  plt.axis('equal')
  display_floor(floor.cpu())
  
  break


#%%

  # for i, floor in enumerate(test_floors):
  #   print(i)
  #   plt.axis('equal')
  #   display_floor(floor)
  #   plt.show()
    
  # #%%
  # display_floor(train_floors[18])
  # # %%
  # k = 2
  # rand_points = test_points[k][np.random.choice(test_points[k].shape[0], 1000)]
  # plt.scatter(*rand_points[:, :2].T)
  # display_floor(test_floors[k])
  # #%%

  # floor = train_floors[np.array([len(x) for x in train_floors]).argmax()]
  # floor = floor.reshape(-1,2,2)
  # lens = np.linalg.norm(floor[:,0] - floor[:,1], axis=1)

  # # sort floor by lens
  # floor = floor[np.argsort(lens)].reshape(-1,4)

  # for edge in floor[50:]:
  #   plt.plot(edge[::2], edge[1::2])


  # # %%