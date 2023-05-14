# Pytorch

### 模型的保存

```python
vgg16 = torchvision.models.vgg15(pretrained=False)
# 保存方式1 模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2 模型参数(官方推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
```



### 模型的加载

```python
# 加载方式1
model = torch.load("vgg16_method1.pth")
print(model)

# 加载方式2
model = torch.load("vgg16_method2.pth")
print(model) # 输出参数，无网络模型结构
model = torchvision.models.vgg16(pretrained=False)
model.load_state_dict(torch.load("vgg16_method2.pth"))
print(model) # 输出参数，有网络结构
```



### 完整的模型训练套路(以CIFAR10数据集)

```python
# train.py
# 准备数据集，数据集分为训练集和测试集
# 全部流程：准备数据集，加载数据集，准备模型，设置损失函数，设置优化器，开始训练，没训练完一轮进行一次测试，结果用tensorboard.SummaryWriter展示，保存模型

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

# 检查数据集的个数
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)


# 创建网络模型
NET = net()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2 #科学计数法
optimzer = torch.optim.SGD(NET.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 10


# 添加tensorboard
writer = SummaryWriter("../logs_train")

start_time = time.time()

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    
    # 训练步骤开始
    NET.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = NET(imgs)
        loss = loss_fn(outputs, targets)
        
        # 优化器模型,梯度清零，反向传播，参数优化
        optimzer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练次数：{}耗时：{} s".format(total_train_step, end_time - start_time)
        	print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
        	writer.add_scalar("total_train_loss", loss.item(), total_train_step)
        
    
    # 每训练完一轮后，在测试数据集上评估一下模型的训练的如何，包括整体测试误差（每一轮应该下降）和整体测试精度（每一轮应该上升）
    NET.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 这里的参数没有梯度，不会对模型进行更改调优
        for data in test_dataloader:
            imgs, targets = data
        	outputs = NET(imgs)
        	loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_accuracy = (outputs.argmax(1) == targets).sum() / test_data_size
    total_test_step = i
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy))
    writer.add_scalar("total_test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_accuracy", total_accuracy, total_test_step)
    
    torch.save(NET, "NET_{}.pth".format(i+1))
    # 官方推荐的保存方式
    torch.save(NET.state_dict(), "NET_{}.pth".format(i+1))
    print("模型已保存")

writer.close()



# model.py
#搭建神经网络
class net(nn.Module):
    self.__init__(self):
        super(net, self).__init__()
        self.model = nn.Sequential(……)
        
	def forward(self, x):
        output = self.model(x)
        return output
    
# 程序主函数
if __name__ == '__main__'
	NET = net()
    input = torch.ones((64, 3, 32, 32))
    output = NET(input)
    print(output.shape)

```



### GPU训练

```python
# 网络模型、损失函数、数据（imgs，targets）调用GPU

if torch.cuda.is_available():
    NET = NET.cuda()
# or
NET = NET.to(device)

```



### 完整的模型验证套路(测试，demo)-利用已经训练好的模型，然后给它提供输入

```python
# 加载一个图片

image_path = ""
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.resize((32, 32)), torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

model = torch.load("", map_location=torch.device("cpu"))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1)) #直观的分类
```



### PS:了解Github



