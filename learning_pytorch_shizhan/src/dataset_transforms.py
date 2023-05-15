import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

trans_totensor = transforms.ToTensor()
# trans_random = transforms.RandomCrop((12, 12))
# dataset_transforms = transforms.Compose([trans_random, trans_totensor])
dataset_transforms = transforms.Compose([trans_totensor])

train_set = torchvision.datasets.CIFAR10(root="../datasets", train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=dataset_transforms, download=True)

# print(test_set[0])
# img, target = test_set[0]
# img.show()

writer = SummaryWriter("../datasets.logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("CIFAR10_formal", img, i)


writer.close()