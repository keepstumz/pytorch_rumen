# 神经网络中使用卷积层 convolution layer
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from torchvision.utils import make_grid

# 1.选取数据集
dataset = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=transforms.ToTensor(), download=True)

# 2.加载数据集
dataloader = DataLoader(dataset=dataset, batch_size=64)


# 3.构建模型 convolution layer：过滤器捕获数据中存在的空间模式或特征，例如边缘、纹理、形状或更高级别的结构
class conv_model(nn.Module):
    def __init__(self):
        super(conv_model, self).__init__()
        # convolution layer 1
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=1, padding=0)

    # 前向传播函数
    def forward(self, x):
        output = self.conv1(x)
        return output


# 4.初始化网络利用写好的conv_model
CONV_MODEL = conv_model() #这个就是网络模型
# print(CONV_MODEL)

writer = SummaryWriter(log_dir="../dataloader.logs")
step = 0


# 5.将加载好的数据输入模型
for data in dataloader:
    imgs, targets = data
    output = CONV_MODEL(imgs)
    print(imgs.shape)
    print(output.shape)

    # torch.Size([64, 3, 32, 32])
    writer.add_images("input_imgs", imgs, step)
    # torch.Size([64, 6, 30, 30])  ---> [xxx, 3, 30, 30]
    output_final = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output_imgs", output_final, step)
    step += 1


writer.close()

