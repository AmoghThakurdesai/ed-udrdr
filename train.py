from torch.utils.data import DataLoader
from torchvision import transforms
from data import CustomDataset
import torch
import torch.nn as nn
import torch.optim as Adam

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = CustomDataset(root_dir='path_to_train_data', transform=transform)
test_dataset = CustomDataset(root_dir='path_to_test_data', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Now you can use these loaders in your training and testing loops
# Here is a simple example of how to use them in a training loop
num_epochs = 10

optimizer = Adam(model.parameters(), lr=0.001)

# Assuming you have a loss function named 'criterion'
criterion = nn.MSELoss()


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
for epoch in range(num_epochs):
    for i, images in enumerate(train_loader):
        # Move images to the device
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, images)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')