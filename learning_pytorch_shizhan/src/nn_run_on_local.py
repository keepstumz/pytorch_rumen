# -*- coding: utf-8 -*-
# Author: _mu_
# time: 2023.

# 准备数据集，数据集分为训练集和测试集
# 全部流程：准备数据集，加载数据集，准备模型，设置损失函数，设置优化器，开始训练，没训练完一轮进行一次测试，结果用tensorboard.SummaryWriter展示，保存模型
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
# from torch.nn.modules.flatten import Flatten
import time
from torch import nn
from torch.utils.data import DataLoader
from nn_sequential import net

# 定义训练设备
device = torch.device("cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../datasets", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

# 检查数据集的个数
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)


# class net(nn.Module):
#     def __init__(self):
#         super(net, self).__init__()
#         # self.conv1 = Conv2d(3, 32, 5, padding=2)
#         # self.maxpool1 = MaxPool2d(2)
#         # self.conv2 = Conv2d(32, 32, 5, padding=2)
#         # self.maxpool2= MaxPool2d(2)
#         # self.conv3 = Conv2d(32, 64, 5, padding=2)
#         # self.maxpool3 = MaxPool2d(2)
#         # self.flatten = Flatten()
#         # self.linear1 = Linear(1024, 64)
#         # self.linear2 = Linear(64, 10)
#         self.sequential = Sequential(Conv2d(3, 32, 5, padding=2),
#                                      ReLU(),
#                                      BatchNorm2d(32),
#                                      MaxPool2d(2),
#                                      Conv2d(32, 32, 5, padding=2),
#                                      ReLU(),
#                                      BatchNorm2d(32),
#                                      MaxPool2d(2),
#                                      Conv2d(32, 64, 5, padding=2),
#                                      BatchNorm2d(64),
#                                      ReLU(),
#                                      MaxPool2d(2),
#                                      Flatten(),
#                                      Linear(1024, 64),
#                                      Linear(64, 10))
#
#     def forward(self, x):
#         # x = self.conv1(x)
#         # x = self.maxpool1(x)
#         # x = self.conv2(x)
#         # x = self.maxpool2(x)
#         # x = self.conv3(x)
#         # x = self.maxpool3(x)
#         # x = self.flatten(x)
#         # x = self.linear1(x)
#         # x = self.linear2(x)
#         x = self.sequential(x)
#         output = x
#         return output


# 创建网络模型
NET = net().to(device)

# 损失函数
# loss_fn = nn.CrossEntropyLoss().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
learning_rate = 1e-2  # 科学计数法
optimzer = torch.optim.SGD(NET.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 100

# 添加tensorboard
writer = SummaryWriter("../logs_train")

start_time = time.time()

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))
    total_accuracy_train = 0
    # 训练步骤开始
    NET.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = NET(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器模型,梯度清零，反向传播，参数优化
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        # 训练集上的准确率
        accuracy_train = (outputs.argmax(1) == targets).sum()
        total_accuracy_train += accuracy_train.item()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练次数：{}耗时：{} s".format(total_train_step, end_time - start_time))
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("total_train_loss", loss.item(), total_train_step)

    print("整体训练集上的准确率：{}".format(total_accuracy_train / train_data_size))

    # 每训练完一轮后，在测试数据集上评估一下模型的训练的如何，包括整体测试误差（每一轮应该下降）和整体测试精度（每一轮应该上升）
    NET.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 这里的参数没有梯度，不会对模型进行更改调优
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = NET(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()
    total_test_step = i
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(float(total_accuracy / test_data_size)))
    writer.add_scalar("total_test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_accuracy", total_accuracy, total_test_step)

    # torch.save(NET, "NET_{}.pth".format(i + 1))
    # 官方推荐的保存方式
    torch.save(NET.state_dict(), "../trained_save/NET_{}_local.pth".format(i + 1))
    print("模型已保存")

writer.close()
