# -*- coding: utf-8 -*-
# Author: _mu_
# time: 2023.05.11

import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# 需要展示的数据集
dataset = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=transforms.ToTensor(), download=True)

# 加载数据集
dataloader = DataLoader(dataset=dataset, batch_size=64)

input = torch.tensor([[0.17324482, 0.36729364, 0.84591261, 0.61793138, 0.67105541],
                      [0.85698073, 0.25400122, 0.90815534, 0.60365728, 0.15738224],
                      [0.61467116, 0.04102846, 0.73522952, 0.76097973, 0.96742777],
                      [0.80233513, 0.37794291, 0.64010461, 0.95688161, 0.39035807],
                      [0.21736549, 0.98975948, 0.78982957, 0.20969979, 0.43994766]], dtype=torch.float32)
# reshape
input = torch.reshape(input, [-1, 1, 5, 5])


# 建立模型 maxpooling是为了减少计算量，降低数据维度
class pool_model(nn.Module):
    def __init__(self):
        super(pool_model, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool(x)
        return output


writer = SummaryWriter("../dataloader.logs")

# 实例化模型
POOL_MODEL = pool_model()
output = POOL_MODEL.maxpool(input)
print(output)

step = 0
# 输入模型
for data in dataloader:
    imgs, targets = data
    output = POOL_MODEL(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("imgs_input_pooling", imgs, step)
    # torch.Size([64, 3, 11, 11])
    writer.add_images("imgs_output_pooling", output, step)
    step += 1

writer.close()