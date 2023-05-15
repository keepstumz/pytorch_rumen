from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    """构造函数需要根目录地址和标签地址"""

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_path_list = os.listdir(os.path.join(self.root_dir, self.label_dir))  # 将两个路径进行合并，并将合并之后的文件夹里的内容以列表形式存储

    """获取图像文件夹下的各个图片，返回值是图像和其标签"""

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)  # Image.open需要知道图像的具体路径
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path_list)


root_dir = "datasets/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
print(len(ants_dataset))
print(len(bees_dataset))
train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))

img, label = train_dataset[200]
print(label)
img.show()
