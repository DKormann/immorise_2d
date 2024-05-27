#%%
import torch
from floor_decoder.data import dataset
import matplotlib.pyplot as plt

#%%

room = dataset[0]
plt.plot(room[:, 0], room[:, 1])



