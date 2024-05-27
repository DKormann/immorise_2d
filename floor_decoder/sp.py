#%%
import torch
import matplotlib.pyplot as plt


w = 3
x = torch.linspace(-w,w, 100)

y = ((-x.abs()).exp() + x + x.abs())/2
b = 1.4
sp = 1/b*(1+(b*x).exp()).log()

plt.plot(x, y)
plt.plot(x,sp)

#%%

yd = -x.sign() * (-x.abs()).exp() + 1 + x.sign()
spd = 1 / (1+(-x).exp()) * 2
plt.plot(x, yd)
plt.plot(x,spd)
