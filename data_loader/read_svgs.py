#%% 
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import numpy as np
import torch

#%%
datapath = os.path.expanduser("/shared/datasets/ImagesGT/")

proj_files = os.listdir(datapath)
floors = []

for proj in sorted(proj_files):
  with open(datapath+proj, 'r') as f:
    floor = []
    for line in f:
      if "class=\"Room" in line:
        line = line.split("points=\"" )[1].split("\"")[0]
        room = [point.split(",") for point in line.split(" ") if point != '']
        floor.append(np.array(room, dtype=np.float32).reshape(-1, 2))
    floors.append(floor)

floors = [[room - np.min([room.min(0) for room in floor], axis=0) for room in floor] for floor in floors]
floors = [[room / np.max([room.max() for room in floor],0) for room in floor] for floor in floors]
np.max([room.max(0) for floor in floors for room in floor],0)

#%%
imsize = np.array([1000,1000])

def transform(floor):
  randpad = np.random.rand(2) * (1-np.max([room.max(0) for room in floor],0))
  floor = [np.float32(randpad) + room for room in floor]
  x_swap,y_swap,ax_swap = np.random.rand(3) > 0.5
  for room in floor: 
    if ax_swap: room[:] = room[::-1,[1,0]]
    if x_swap: room[:,0] = 1 - room[:,0]
    if y_swap: room[:,1] = 1 - room[:,1]
    room_top = room[:].mean(1).argmax()
    room = np.concatenate([room[:room_top], room[room_top:]])

  floor = sorted(floor, key=lambda x: x[:,1].mean())
  return floor

#%%


def rasterize(floor):
  image = Image.new("L", (*imsize,),0)
  for polygon in floor: 
    ImageDraw.Draw(image).polygon(polygon.copy() * 1000, fill=1)
  # image = image.rotate(np.random.randint(0,4)*90, expand=True)
  return np.array(image)

train_floors, test_floors = floors[:100], floors[100:]
max_edges = 102
n_toks = 100



def get_edges(floors):
  edges = []
  for floor in floors:
    floor = [np.concatenate([room, room[:1]], axis=0) for room in floor]
    floor_edges = np.concatenate([np.concatenate([room[:-1],room[1:]],axis=1) for room in floor])
    floor_edges = floor_edges * (n_toks-1) + 1
    floor_edges = floor_edges.astype(np.int32).clip(1, n_toks-1)
    floor_edges = np.pad(floor_edges, ((0, max_edges - len(floor_edges)), (0, 0)))
    edges.append(floor_edges)
  return np.stack(edges)


import torch 

def get_batch(floors):
  floors = [transform(floor) for floor in floors]
  images = np.stack([rasterize(floor) for floor in floors])
  edges = get_edges(floors).reshape(-1, max_edges*4)
  return torch.tensor(images), torch.tensor(edges)

get_train_batch = lambda: get_batch(train_floors)
get_test_batch = lambda: get_batch(test_floors)


#%%

if __name__ == "__main__":
  img, edg = get_train_batch()
  print(img.shape, edg.shape)

