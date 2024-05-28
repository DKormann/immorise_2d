#%%
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
datapath = os.path.expanduser("~/datasets/v2")

proj_files = os.listdir(datapath)
floors = []
dataset = []
for proj in proj_files:
    proj_path = os.path.join(datapath, proj)
    files = os.listdir(proj_path)
    for file in files:
      floor_rep = {"name": file, "rooms": 0}
      with open(os.path.join(proj_path, file), 'r') as f:
        for line in f:          
          words  = line.split()
          name, data= ' '.join(words[:-1]), words [-1]

          floor_rep['rooms'] += 1
          data = data.split('_') [1:]
          if data == ['']: continue
          try:
            data = np.array(data, dtype=np.float32).reshape(-1, 2)
            dataset.append(data)
            data = np.concatenate([data, data[0:1]])
          except: print('couldnt parse', data)
      floors.append(floor_rep)
#%%