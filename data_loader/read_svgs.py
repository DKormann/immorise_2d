#%% 
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
datapath = os.path.expanduser("~/datasets/ImagesGT/")

proj_files = os.listdir(datapath)
floors = []
dataset = []
for proj in proj_files:

  with open(datapath+proj, 'r') as f:
    for line in f:
      if "class=\"Room" in line:

        line = line.split("points=\"" )[1].split("\"")[0]
        points = line.split(" ")
        data = np.array(points, dtype=np.float32).reshape(-1, 2)

        print(data,end='\n')

  break
    

