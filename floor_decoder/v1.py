#%%
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
datapath = os.path.expanduser("~/datasets/floors/dec_data")

proj_files = os.listdir(datapath)
for proj in proj_files:
    print("\n", proj)
    proj_path = os.path.join(datapath, proj)
    files = os.listdir(proj_path)
    for file in files:
      print('\n',file)
      plt.axes().set_aspect('equal', 'datalim')
      with open(os.path.join(proj_path, file), 'r') as f:
        for line in f:
          
          words  = line.split()
          name, data= ' '.join(words[:-1]), words [-1]
          # print(name)
          

          data = data.split('_') [1:]
          if data == ['']: continue
          try:
            data = np.array(data, dtype=np.float32).reshape(-1, 2)
            print(data.shape[0], end = ' ')
            data = np.concatenate([data, data[0:1]])
            plt.plot(data[:,0], data[:,1])
          except:
            print('couldnt parse', data)
          # print(line, end='')
      plt.show()
      
