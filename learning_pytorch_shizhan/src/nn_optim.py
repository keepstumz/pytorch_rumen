# -*- coding: utf-8 -*-
# Author: _mu_
# time: 2023.


import torch
from torch import optim
import torch.nn as nn
import torchvision.datasets
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
from torch.nn.modules.flatten import Flatten
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


dataset = torchvision.datasets.CIFAR10(root="../datasets", train=False,transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


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
                                     MaxPool2d(2),
                                     Conv2d(32, 32, 5, padding=2),
                                     MaxPool2d(2),
                                     Conv2d(32, 64, 5, padding=2),
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
        x =self.sequential(x)
        output = x
        return output


# 实例化网络模型，选择损失函数，设置优化器 running on CPU
NET = net()
Optim = optim.SGD(NET.parameters(), lr=0.01)
loss_cross = nn.CrossEntropyLoss()
for epoch in range(20):
    result_running = 0
    for data in dataloader:
        imgs, targets = data
        outputs = NET(imgs)
        Optim.zero_grad()
        # print(outputs)
        # print(targets)
        result_loss_cross = loss_cross(outputs, targets)
        # print(result_loss_cross)
        # 反向传播，得到调整后的参数
        result_loss_cross.backward()
        Optim.step()
        result_running += result_loss_cross
    print(result_running)


# # 实例化网络模型，选择损失函数，设置优化器 running on GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# NET = net().to(device)
# Optim = optim.SGD(NET.parameters(), lr=0.01)
# loss_cross = nn.CrossEntropyLoss().to(device)
# for epoch in range(20):
#     result_running = 0
#     for data in dataloader:
#         imgs, targets = data
#         imgs = imgs.to(device)
#         targets = targets.to(device)
#         outputs = NET(imgs)
#         Optim.zero_grad()
#         # print(outputs)
#         # print(targets)
#         result_loss_cross = loss_cross(outputs, targets)
#         # print(result_loss_cross)
#         # 反向传播，得到调整后的参数
#         result_loss_cross.backward()
#         Optim.step()
#         result_running += result_loss_cross
#     print(result_running)

