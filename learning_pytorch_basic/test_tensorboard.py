from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")  # 创建一个logs文件夹，writer写的文件都在该文件夹下
"""写完的日志文件存储在logs文件夹中，需要读取时在终端使用 tensorboard --logdir=logs --port=6008,自定义端口号，避免重复占用"""

for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

"""导入图片进入tensorboard中进行显示"""
img_path_ants = "datasets/train/ants/0013035.jpg"
img_ants = Image.open(img_path_ants)
img_ants_np_array = np.array(img_ants)

print(img_ants_np_array.shape)

img_path_bees = "datasets/train/bees/16838648_415acd9e3f.jpg"
img_bees = Image.open(img_path_bees)
img_bees_np_array = np.array(img_bees)

writer.add_image("img", img_ants_np_array, 1, dataformats='HWC')
writer.add_image("img", img_bees_np_array, 2, dataformats='HWC')

writer.close()
# help(SummaryWriter)
