
# 看神经网络官方文档进行简单的神经网络学习
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义一个张量
x = torch.tensor(1.0)


class SUM(nn.Module):
    def __init__(self):
        super().__init__() # 类的初始化函数

    def forward(self, input):
        output = input + 1 # 前向传播
        return output


sum = SUM() # 实例化类
output = sum(x)
print(output)