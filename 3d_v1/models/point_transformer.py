#%%

import torch
import utils.data_2d




#%%

class PointTransformer(torch.nn.Module):
  def __init__(self):
    super(PointTransformer, self).__init__()
