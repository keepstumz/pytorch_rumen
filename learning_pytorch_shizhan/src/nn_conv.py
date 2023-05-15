import torch
import torch.nn.functional as F
# 输入一个张量
input = torch.tensor([[0.17324482, 0.36729364, 0.84591261, 0.61793138, 0.67105541],
 [0.85698073, 0.25400122, 0.90815534, 0.60365728, 0.15738224],
 [0.61467116, 0.04102846, 0.73522952, 0.76097973, 0.96742777],
 [0.80233513, 0.37794291, 0.64010461, 0.95688161, 0.39035807],
 [0.21736549, 0.98975948, 0.78982957, 0.20969979, 0.43994766]])

# 自定义一个filter（过滤器）
kernel = torch.tensor([[0.17324482, 0.36729364, 0.84591261],
 [0.61793138, 0.67105541, 0.85698073],
 [0.25400122, 0.90815534, 0.60365728]])
# print(input.shape)
# print(kernel.shape)
# reshape改变维数

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# 卷积层的计算
# 1.最简单的卷积层操作
output = F.conv2d(input, kernel, stride=1)
print(output)
# 2.改变步长
output2 = F.conv2d(input, kernel, stride=2)
print(output2)
# 3.增加填充
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
