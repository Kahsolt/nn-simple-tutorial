#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from model import Model

import torch
from torch.nn import Module
import torch.nn.functional as F

BATCH_SIZE = 32

model: Module = Model(1, 1)
state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)


X = torch.Tensor([1.5])
Y_hat = model(X)
print('pred:', Y_hat)
