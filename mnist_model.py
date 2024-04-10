#%%

import torch 
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt



# Download and load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
#%%

# Convert train data into torch tensors
train_data = train_dataset.data
train_targets = train_dataset.targets

train_data = train_data.unsqueeze(1).float() / 255.0
train_targets = torch.tensor(train_targets)


#%%

i=5
print(train_targets[i])
plt.imshow(train_data[i,0], cmap='gray')
#%%

class Model(torch.nn.Module):
  
  def __init__(self):
    super().__init__()
    self.W = torch.randn(784, 10) / 28
    self.b = torch.zeros(10)
    
  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = torch.matmul(x, self.W) + self.b
    return x

model = Model()
opt = torch.optim.SGD([model.b, model.W], lr=1e-2)
#%%
prediction = model(train_data)

def loss_fn(p, y):
  p = p.softmax(dim=1)
  l = F.cross_entropy(p, y)
  return l


l = loss_fn(prediction, train_targets)

opt.zero_grad()
# l.backward()
# opt.step()
