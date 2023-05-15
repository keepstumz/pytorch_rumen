import torchvision

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("../dataloader.logs")
# 准备测试数据集，将其预处理为张量形式
test_data = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=transforms.ToTensor())
# 加载数据，每一步处理的个数，顺序是否打乱
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# 测试数据集中的第一张图片及target
# img, target = test_data[0]
# print(img, target)

# epoch 一般将过程中每一步展示出来
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images('dataloader_Images', imgs, global_step=step)
        writer.add_images('Epoch = {}'.format(epoch), imgs, global_step=step)
        step += 1


writer.close()

# # chatgpt 给出的标准代码
# import torchvision
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter("dataloader.logs")
#
# # Prepare the test dataset
# test_data = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=transforms.ToTensor())
#
# test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
#
# # Access the first image and target
# img, target = test_data[0]
# print(img.shape, target)
#
# step = 0
#
# for data in test_loader:
#     imgs, targets = data
#     print(imgs.shape)
#     print(targets)
#
#     # Convert the image tensor to a grid
#     grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True)
#
#     # Log the grid to TensorBoard
#     writer.add_image('dataloader_Images_chatgpt', grid, global_step=step)
#     step += 1
#
# writer.close()
