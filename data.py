#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from typing import Tuple
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class LambdaDataset(Dataset):
  
  def __init__(self):
    super().__init__()

    self.func = lambda x: x + 1

  def __len__(self):
    return 10000

  def __getitem__(self, idx:int) -> Tuple[float, float]:
    x = random.random() * 1000 - 500    # [-500, 500]
    y = self.func(x)
    return np.asarray([x]).astype(np.float32), np.asarray([y]).astype(np.float32)
