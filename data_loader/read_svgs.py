#%% 
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import numpy as np

#%%
datapath = os.path.expanduser("~/datasets/ImagesGT/")

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
fmax = np.max([room.max(0) for floor in floors for room in floor],0)

#%%

imsize = np.array([4000,4000])
floor = floors[1]

#%%


def rasterize(floor):

  fmax = np.max([room.max(0) for room in floor],0)
  imsize - fmax
  rand_pad = np.random.rand(2) * (imsize - fmax)
  floor = [np.float32(rand_pad) + room for room in floor]

  ax_swap = np.random.rand() > 0.5
  x_swap = np.random.rand() > 0.5
  y_swap = np.random.rand() > 0.5
  mirrored = ax_swap ^ x_swap ^ y_swap

  for room in floor: 
    if ax_swap: room[:] = room[:,[1,0]]
    if x_swap: room[:,0] = imsize[0] - room[:,0]
    if y_swap: room[:,1] = imsize[1] - room[:,1]

  image = Image.new("L", (*imsize,),0)
  for polygon in floor: ImageDraw.Draw(image).polygon(polygon.copy(), fill=1)
  return np.array(image)


plt.imshow(pixel_array, cmap='gray')
# %%
