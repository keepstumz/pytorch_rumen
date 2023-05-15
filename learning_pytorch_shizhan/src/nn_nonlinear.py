# -*- coding: utf-8 -*-
# Author: _mu_
# time: 2023.05.11

import torch
import torch.nn as nn
import torchvision.datasets
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

input = torch.tensor([[1, -0.5],
                      [2, -5.3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input)

# 加载数据集

dataset = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset=dataset, batch_size=64)


# 建立模型的经典步骤
class Non_linear(nn.Module):
    def __init__(self):
        super(Non_linear, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # output = self.relu(x)
        output = self.sigmoid(x)
        return output


writer = SummaryWriter("../dataloader.logs")

# 输入数据
NON_LINEAR = Non_linear()

output1 = NON_LINEAR.relu(input)
output2 = NON_LINEAR.sigmoid(input)
print(output1)
print(output2)


step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("imgs_input", imgs, step)
    # output1 = NON_LINEAR(imgs)
    output2 = NON_LINEAR(imgs)
    # writer.add_images("imgs_output_relu", output1, step)
    writer.add_images("imgs_output_sigmoid", output2, step)
    step += 1

writer.close()
