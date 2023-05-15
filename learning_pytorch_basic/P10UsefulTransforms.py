from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

writer = SummaryWriter("transforms_logs")

# img_PIL = Image.open(r"D:\pythonProject\learning_pytorch\datasets\train\ants\69639610_95e0de17aa.jpg")
# image = img_PIL.convert("RGB") # 图片转换png格式为jpg格式
# image.save(r"D:\pythonProject\learning_pytorch\datasets\train\yellow.jpg")

image_PIL = Image.open(r"D:\pythonProject\learning_pytorch\datasets\train\ants\69639610_95e0de17aa.jpg")

# transforms包中常用的
# 1.ToTensor使用
trans_totensor = transforms.ToTensor()
img_totensor = trans_totensor(image_PIL)

writer.add_image("ToTensor_img", img_totensor)
# print(trans_totensor[0][0][0])
# 1.2 ToPILimage使用
trans_toPILimage = transforms.ToPILImage()(img_totensor)
# trans_toPILimage.show()
# writer.add_image("ToPIL_img", trans_toPILimage)

# 2.Normalize使用  ``input[channel] = (input[channel] - mean[channel]) / std[channel]``正则化公式
trans_normalize = transforms.Normalize([.06, .3, .5], [0.50, 0.50, 0.02])(img_totensor)
writer.add_image("Normalize_img", trans_normalize, 8)
# print(trans_normalize[0][0][0])

# Resize
# 1.输入一个sequence进行resize方法
trans_resize = transforms.Resize((512, 512))(image_PIL)
trans_resize_img = transforms.ToTensor()(trans_resize)
# print(trans_resize_img)
writer.add_image("Resize_img", trans_resize_img, 0)

# 2.输入一个int进行等比resize方法 compose方法是多种transforms整合在一起执行的操作
# compose中，前面的输出作为后面的输入，注意格式匹配
# PIL -> PIL -> Tensor
trans_resize_2 = transforms.Resize(1024)
trans_resize_img_2 = transforms.Compose([trans_resize_2, trans_totensor])(image_PIL)
# print(trans_resize_img_2)
writer.add_image("Resize_img", trans_resize_img_2, 1)

#  3.RandomCrop 随机裁剪，举个例子学会包里面的任意类用法
trans_random = transforms.RandomCrop((30, 200))
for i in range(10):
    img_random = transforms.Compose([trans_random, trans_totensor])(image_PIL)
    writer.add_image("RandomCropHW", img_random, i)


writer.close()

