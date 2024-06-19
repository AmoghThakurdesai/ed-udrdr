from torch.utils.data import DataLoader
from torchvision import transforms
from data import CustomDataset
import torch
import torch.nn as nn
from torch.optim import Adam
from model import *

# TODO: Modify the train loader so that the real images can be added alongside the training images


def train_model(model, train_loader, num_epochs, device):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for i, (rainy_images, clear_images) in enumerate(train_loader):
            rainy_images = rainy_images.to(device)
            clear_images = clear_images.to(device)

            # Forward pass
            outputs = model(rainy_images)

            # Compute loss
            loss = criterion(outputs, clear_images)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')


def train_big_model(train_loader,device,num_epochs):
    print(f'Before:{torch.cuda.memory_allocated()}')
    
    REN = RainEstimationNetwork().to(device)

    print(f'After REN:{torch.cuda.memory_allocated()}')
    RCN = RainEstimationNetwork().to(device)
    print(f'After RCN:{torch.cuda.memory_allocated()}')
    DN = RainEstimationNetwork().to(device)
    print(f'After DN:{torch.cuda.memory_allocated()}')
    criterion1 = nn.L1Loss()
    # Define the optimizer
    optimizer = Adam(list(REN.parameters()) + list(RCN.parameters()) + list(DN.parameters()))), lr=0.001)

    for epoch in range(num_epochs):
        for i, (x_syn, y_gt, x_real) in enumerate(train_loader):
            x_syn = x_syn.to(device)
            y_gt = y_gt.to(device)
            z_gt = x_syn - y_gt
            z_gt = z_gt.to(device) 
            x_real = x_real.to(device)

            print(f'{torch.cuda.memory_allocated() * 4 / (1024 ** 3)}')
            # Forward pass
            z_syn = REN(x_syn)
            z_real = REN(x_real)
            
            loss1 = criterion1(z_gt,z_syn)

            rcn_input_syn = x_syn - z_syn
            rcn_input_real = x_real - z_real
            y_syn = RCN(rcn_input_syn)
            y_real = RCN(rcn_input_real)
            loss2 = criterion1(y_syn,y_gt)

            
            x_ref = z_real + y_gt

            dn_output_ref = DN(x_ref)
            dn_output_real = DN(x_real)

            loss3 = criterion1(dn_output_real,y_real) + criterion1(dn_output_ref,y_gt)
            loss = loss1 + loss2 + loss3


            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

            # Save the model
    # Save the models
    torch.save({
            'REN_state_dict': REN.state_dict(),
            'RCN_state_dict': RCN.state_dict(),
            'DN_state_dict': DN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            }, 'models.pth')

 
            




device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)
# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = CustomDataset(root_dir='/content/rainy-image-dataset', image_dir='rainy image', label_dir='ground truth', real_image_dir="/content/drive/MyDrive/Machine_Learning/Paper/Custom/GT-RAIN_train",num_samples=10000,transform=transform)


# Create data loaders.
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


# Train and test the model
train_big_model(train_loader, num_epochs=1, device=device)

