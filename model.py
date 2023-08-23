#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Linear
from torch.nn import ReLU


class Model(Module):
    
  def __init__(self, d_in:int, d_out:int):
    super().__init__()

    self.net = Sequential(
      Linear(d_in, 2 * d_in),   # [B, D=6]
      ReLU(),                   # [B, D=6]
      Linear(2 * d_in, d_out),  # [B, D=1]
    )
  
  def forward(self, x:Tensor) -> Tensor:
    return self.net(x)    # [B, D=3]
