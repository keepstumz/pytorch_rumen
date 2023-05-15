# -*- coding: utf-8 -*-
# Author: _mu_
# time: 2023.05.12

# loss fuction

import torch
from torch import nn


inputs = torch.tensor([1, 1, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 9], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# 各元素相差
loss = nn.L1Loss(reduction="sum")
result = loss(inputs, targets)
print(result)

# 各元素相差的平方、均方差
lossMSE = nn.MSELoss(reduction="sum")
result_MSE = lossMSE(inputs, targets)
print(result_MSE)

# 交叉熵损失
x = torch.tensor([0.1, 0.2, 0.8]).reshape(1, -1)
y = torch.tensor([2])
loss_cross = nn.CrossEntropyLoss()
result_crossloss = loss_cross(x, y)
print(result_crossloss)





