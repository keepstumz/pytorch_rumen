# -*- coding: utf-8 -*-
# Author: _mu_
# time: 2023.
import torchvision.transforms
from PIL import Image
import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, ReLU, BatchNorm2d
from torch.nn.modules.flatten import Flatten
from torch import nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载待验证的图片
image_pth = "../verify_imgs/cat.jpg"
image = Image.open(image_pth)
print(image)

# 利用transform将图片重塑
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))])
img = transform(image)
img = torch.reshape(img, (1, 3, 32, 32)).to(device)


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


model = net().to(device)
model.load_state_dict(torch.load("../trained_save/NET_200.pth"))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
