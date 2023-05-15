import torch
import torchvision.transforms as transforms
from PIL import Image
from MNIST_train import SimpleNet

# Load the trained model
model = torch.load('MNIST_trained.pth')
# model = SimpleNet().load_state_dict(torch.load('MNIST_100_trained.pth'))

# Move the model to the desired device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize the image to match the input size of the model
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image data
])

# Load and preprocess the image
image = Image.open(r'C:\Users\Administrator\Desktop\number.jpg')
image = transform(image).to(device)  # Add a batch dimension
# image = transform(image).unsqueeze(0).to(device)  # Add a batch dimension

# # Flatten the image tensor
# image = image.view(image.size(0), -1)

# Make predictions
with torch.no_grad():
    output = model(image)

# Get the predicted class
_, predicted = torch.max(output, 1)
predicted_class = predicted[0].item() # Use the item() method on the first element

print("Predicted class:", predicted_class)
