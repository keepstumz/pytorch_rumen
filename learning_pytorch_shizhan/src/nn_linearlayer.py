# -*- coding: utf-8 -*-
# Author: _mu_
# time: 2023.05.11

import torch
import torch.nn as nn

# 加载数据集
import torchvision.datasets
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

dataset = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset=dataset, batch_size=64, drop_last=True)  # drop_last参数默认为False,不会丢弃最后的不和尺寸的数据


# 建立模型 linear layer 指定输入特征数和输出特征数和输出特征数
class linear(nn.Module):
    def __init__(self):
        super(linear, self).__init__()
        self.linear = Linear(196608, 30)

    def forward(self, x):
        output = self.linear(x)
        return output


LINEAR = linear()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # imgs = torch.reshape(imgs, (1, 1, 1, -1))
    imgs = torch.flatten(imgs)
    print(imgs.shape)
    output = LINEAR.linear(imgs)
    print(output.shape)
