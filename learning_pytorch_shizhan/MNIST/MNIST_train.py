import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Define the neural network architecture


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input size: 28x28=784, Output size: 128
        self.fc2 = nn.Linear(128, 10)  # Output size: 10 (number of classes)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Set device
device = torch.device("cuda")

# Load and preprocess the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Create the neural network instance
net = SimpleNet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
start_time = time.time()
epoch = 100
for epoch in range(epoch):  # Run for epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            end_time = time.time()
            print("耗时：{}".format(end_time - start_time))
    if epoch == 99:
        torch.save(net.state_dict(), "./MNIST_{}_trained.pth".format(epoch+1))
        print("model is saved successfully!")


print("Training finished!")

# Evaluate the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test images: %.2f %%' % (100 * correct / total))
