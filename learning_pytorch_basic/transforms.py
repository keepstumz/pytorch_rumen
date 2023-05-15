from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

# 1.transforms该如何使用

# 绝对路径：D:\pythonProject\learning_pytorch\datasets\train\ants\0013035.jpg
# 相对路径：datasets/train/ants/0013035.jpg

img_path = "datasets/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# tensor_trans = transforms.ToTensor()  # 初始化
# tensor_img = tensor_trans(img)  # 实例化

tensor_img = transforms.ToTensor()(img)
# print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

writer.close()

# 2.为什么要用tensor数据

cv_img = cv2.imread(img_path)
# print(cv_img)
tensor_img_cv = transforms.ToTensor()(cv_img)

writer.add_image("Tensor_img_cv", tensor_img_cv)

writer.close()

