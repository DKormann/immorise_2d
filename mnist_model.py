#%%
import torch 
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt

from trainplot import TrainPlot

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
    self.W = torch.nn.Parameter(torch.randn(784, 10) / 28)
    self.b = torch.nn.Parameter(torch.zeros(10))
    
  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = torch.matmul(x, self.W) + self.b
    return x

model = Model()
opt = torch.optim.Adam([model.b, model.W], lr=1e-2)
#%%

def loss_fn(p, y):
  p = p.softmax(dim=1)
  l = F.cross_entropy(p, y)
  return l

#%%
def step():
  prediction = model(train_data)
  l = loss_fn(prediction, train_targets)
  opt.zero_grad()
  l.backward()
  opt.step()
  return l.item()

#%%
for i in range(100):
  print(step())
  
#%%

prediction = model(train_data)
#%%
k = 4
plt.plot(prediction[k].softmax(-1).detach())
plt.show()
plt.imshow(train_data[k,0])