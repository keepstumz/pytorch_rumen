# -*- coding: utf-8 -*-
# Author: _mu_
# time: 2023.05.11

import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, ReLU, BatchNorm2d
from torch.nn.modules.flatten import Flatten
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms


# 建立网络模型 easy
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2= MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)
        self.sequential = Sequential(Conv2d(3, 32, 5, padding=2),
                                     ReLU(),
                                     BatchNorm2d(32),
                                     MaxPool2d(2),
                                     Conv2d(32, 32, 5, padding=2),
                                     ReLU(),
                                     BatchNorm2d(32),
                                     MaxPool2d(2),
                                     Conv2d(32, 64, 5, padding=2),
                                     BatchNorm2d(64),
                                     ReLU(),
                                     MaxPool2d(2),
                                     Flatten(),
                                     Linear(1024, 64),
                                     Linear(64, 10))

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.sequential(x)
        output = x
        return output


writer = SummaryWriter("../dataloader.logs")

NET = net()
input = torch.ones(64, 3, 32, 32)
output = NET(input)
# print(output.shape)
writer.add_graph(NET, input)

writer.close()

