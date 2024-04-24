from torch.utils.data import DataLoader
from torchvision import transforms
from data import CustomDataset
import torch
import torch.nn as nn
from torch.optim import Adam
from model import ImageProcessingModel

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = CustomDataset(root_dir='/mnt/d/PythonFiles/assignment/dataset/rainy-image-dataset', image_dir='rainy image', label_dir='ground truth', transform=transform)
test_dataset = CustomDataset(root_dir='/mnt/d/PythonFiles/assignment/ed-udrdr/rainy', image_dir='rainy', label_dir='d rainy', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Now you can use these loaders in your training and testing loops
# Here is a simple example of how to use them in a training loop
num_epochs = 10
model = ImageProcessingModel()
params = model.parameters()
optimizer = Adam(params, lr=0.001)

 

# Assuming you have a loss function named 'criterion'
criterion = nn.MSELoss()


model = ImageProcessingModel()
model = model.to(device)
for epoch in range(num_epochs):
    for i, (rainy_images, clear_images) in enumerate(train_loader):
        
        # Move images to the device
        
        rainy_images = rainy_images.to(device)
        clear_images = clear_images.to(device)
        print("images moved to device")

        # Forward pass
        outputs = model(rainy_images)
        print("forward pass done")
        # Compute loss
        loss = criterion(outputs, clear_images)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')