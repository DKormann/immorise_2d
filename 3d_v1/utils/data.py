#%%
import os
import urllib
import re
import pathlib
# datapath = "../dataset"
datapath = pathlib.Path("/home/service/datasets/immo_v1/")

import numpy as np
import torch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

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

def get_floors(): return map(tokenized, [train_dataset, val_dataset, test_dataset])

#%%

def display_floor(floor, c = 'b'):
  for edge in floor.reshape(-1,4):
    plt.plot(edge[::2], edge[1::2], c)

ntoks = 100

def tokenized(dataset, ntoks = ntoks):
  res = []
  for floor in extract_floors(dataset):
    floor = torch.tensor(floor).view(-1,2,2)
    #normalize
    fmin = floor.min(0)[0].min(0)[0]
    fmax = floor.max(0)[0].max(0)[0]
    maxsize = (fmax - fmin).max()
    floor = (floor - fmin) / maxsize

    #quantize
    floor = (floor * (ntoks-1)).round().long()
    floor = floor.clamp(1,ntoks-1)
    
    res.append(floor)
  return res

