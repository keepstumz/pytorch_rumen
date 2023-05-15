# import torch
#
# # Assuming you have a trained model called 'model'
# # Save the model to a file
# from MNIST import MNIST_train
#
# torch.save(MNIST_train, 'MNIST_trained.pth')


# chatgpt更新过后的code
import torch

# Assuming you have a trained model called 'model'
# Save the model state dictionary to a file
from MNIST.MNIST_train import net

torch.save(net, 'MNIST_trained.pth')
