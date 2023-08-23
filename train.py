#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from data import LambdaDataset, DataLoader
from model import Model

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.optim import Adam

EPOCH = 10000
BATCH_SIZE = 32
LR = 0.01

dataset = LambdaDataset()
dataloader = DataLoader(dataset, BATCH_SIZE)
model: Module = Model(1, 1)
optim = Adam(model.parameters(), lr=LR)

loss_fn = F.mse_loss

finished = False
step = 0
for _ in range(EPOCH):
  if finished: break
  for X, Y in dataloader:
    X: Tensor     # [B=32, D=3], [[23,212,312], ...]
    Y: Tensor     # [B=32],      [[0], [], ...]

    optim.zero_grad()
    Y_hat = model(X)    # forward, [B, D=1]
    loss = loss_fn(Y_hat, Y)
    loss.backward()     # backward
    optim.step()        # 修改 model.parameters

    step += 1

    if step % 10 == 0:
      print(f'step: {step}, loss: {loss.item()}')
      if loss.item() < 1e-8: 
        finished = True
        break

torch.save(model.state_dict(), 'model.pth')
print('Done!')
