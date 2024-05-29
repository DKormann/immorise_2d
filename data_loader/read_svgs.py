#%% 
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import numpy as np

#%%
datapath = os.path.expanduser("/shared/datasets/ImagesGT/")

proj_files = os.listdir(datapath)
floors = []

for proj in proj_files:
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
  randpad = np.random.rand(2) * (np.max([room.max(0) for room in floor],0))
  floor = [np.float32(randpad) + room for room in floor]
  x_swap,y_swap,ax_swap = np.random.rand(3) > 0.5
  for room in floor: 
    if ax_swap: room[:] = room[:,[1,0]]
  return floor


#%%
floor = floors[0]
floor = transform(floor)
floor[0]*imsize
# plt.imshow(rasterize(floor), cmap='gray')

#%%

def rasterize(floor, distortion = False):
  image = Image.new("L", (*imsize,),0)
  for polygon in floor: ImageDraw.Draw(image).polygon(polygon.copy()*imsize, fill=1)
  if distortion: 
    image = image.rotate(np.random.rand()*360, expand=True)
    # TODO: Add distortion
    pass
  return np.array(image)
#%%
def get_edgeset(floor):
  floor = [np.concatenate([room, room[:1]], axis=0) for room in floor]
  edges = []
  for room in floor:
    edges.extend([np.concatenate([room[:-1],room[1:]],axis=1)])
  edges = np.concatenate(edges)
  return edges
edges = [get_edgeset(floor) for floor in floors]

#%%

for i,floor in enumerate(floors):
  image = transform(floor)
  # image=  floor
  image = rasterize(image,False)
  print(i)
  plt.imshow(image, cmap='gray')
  plt.show()