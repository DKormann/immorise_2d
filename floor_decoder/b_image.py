#%%

import os
import torch 
from torch.nn import functional as F
import numpy as np

import matplotlib.pyplot as plt

#%%
datapath = os.path.expanduser("~/datasets/binary_images")
proj_files = os.listdir(datapath)

#%%
c = 16

# for proj in proj_files:
#   path = os.path.join(datapath, proj)


images = []

shapes = []
for proj in proj_files:
  path = os.path.join(datapath, proj)

  raw_images_path = os.path.join(path, "corrected_images")
  raw_edges_path = os.path.join(path, "edges")
  for image_file in os.listdir(raw_images_path):

    edge_file = image_file.replace(".png", ".csv")
    if image_file.endswith(".png"):
      image_path = os.path.join(raw_images_path, image_file)
      images.append(plt.imread(image_path))
    if edge_file.endswith(".csv"):
      shape = []
      edge_path = os.path.join(raw_edges_path, edge_file)
      with open(edge_path, 'r') as f:
        f.readline()
        for line in f.readlines():
          arr = np.array(line.split(',')[2:6], dtype=np.float32)
          shape.append(arr)
      
      shapes.append(np.stack(shape))
  
  
      


assert len(images) == len(shapes)
#%%


def crop_zero(image):
  xsum = image.sum(0)
  ysum = image.sum(1)
  xstart = np.argmax(xsum > 0)
  xend = len(xsum) - np.argmax(xsum[::-1] > 0)
  ystart = np.argmax(ysum > 0)
  yend = len(ysum) - np.argmax(ysum[::-1] > 0)
  return image[ystart:yend, xstart:xend]

images = [crop_zero(i) for i in images]



#%%

sizes = [i.shape for i in images]
xmax,ymax = [max(s) for s in zip(*sizes)]
splits = 8
xmax = xmax // splits * splits
ymax = ymax // splits * splits

idata = torch.stack([F.pad(torch.tensor(i, dtype=torch.float32), (0, xmax-i.shape[1], 0, ymax-i.shape[0])).T for i in images])
idata = idata.reshape(-1, splits, xmax//splits, splits, ymax//splits).permute(0,1,3,2,4)

#%%
