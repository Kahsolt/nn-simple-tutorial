#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from model import Model

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
import torch.nn.functional as F

model: Module = Model(1, 1)
state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)


Xs = torch.linspace(-10000.0, 10000.0, 1000)
Y_hats = []
for x in Xs:
  X = torch.Tensor([x])
  Y_hat = model(X)
  Y_hats.append(Y_hat.item())

plt.plot(Xs, Y_hats)
plt.show()
