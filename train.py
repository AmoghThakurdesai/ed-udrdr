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
    enc = Encoder().to(device)
    dec = Decoder().to(device)

    print(f'After ENC DEC:{torch.cuda.memory_allocated()}')
    criterion1 = nn.L1Loss()
    # Define the optimizer
    optimizer = Adam(list(REN.parameters()) + list(RCN.parameters()) + list(DN.parameters()) + list(enc.parameters()) + list(dec.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        for i, (x_syn, y_gt, x_real) in enumerate(train_loader):
            x_syn = x_syn.to(device)
            y_gt = y_gt.to(device)
            z_gt = x_syn - y_gt
            z_gt = z_gt.to(device) 
            x_real = x_real.to(device)

            print(f'{torch.cuda.memory_allocated() * 4 / (1024 ** 3)}')
            # Forward pass
            x_syn_enc = enc(x_syn)
            x_real_enc = enc(x_real)
            z_syn_enc = REN(x_syn_enc)
            z_real_enc = REN(x_real_enc)
            z_syn = dec(z_syn_enc)
            
            loss1 = criterion1(z_gt,z_syn)

            rcn_input_syn = x_syn_enc - z_syn_enc
            rcn_input_real = x_real_enc - z_real_enc
            y_syn_enc = RCN(rcn_input_syn)
            y_real_enc = RCN(rcn_input_real)
            y_real = dec(y_real_enc)
            y_syn = dec(y_syn_enc)
            loss2 = criterion1(y_syn,y_gt)

            y_gt_enc = enc(y_gt)
            x_ref_enc = z_real_enc + y_gt_enc

            dn_output_ref = DN(x_ref_enc)
            dn_output_real = DN(x_real_enc)

            dn_output_ref_dec = dec(dn_output_ref)
            dn_output_real_dec = dec(dn_output_real)

            loss3 = criterion1(dn_output_real_dec,y_real) + criterion1(dn_output_ref_dec,y_gt)
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
            'enc_state_dict': enc.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            }, 'models.pth')

 
            


def test_model(model, test_loader, device):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    criterion = nn.MSELoss()

    with torch.no_grad():  # No need to track gradients in testing
        for i, (rainy_images, clear_images) in enumerate(test_loader):
            rainy_images = rainy_images.to(device)
            clear_images = clear_images.to(device)

            # Forward pass
            outputs = model(rainy_images)

            # Compute loss
            loss = criterion(outputs, clear_images)

            print(f'Test Step [{i+1}/{len(test_loader)}], Loss: {loss.item()}')

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = CustomDataset(root_dir='/mnt/d/PythonFiles/assignment/dataset/rainy-image-dataset', image_dir='rainy image', label_dir='ground truth', real_image_dir="/mnt/d/PythonFiles/assignment/dataset/GT-RAIN_train",num_samples=30,transform=transform)
test_dataset = CustomDataset(root_dir='/mnt/d/PythonFiles/assignment/ed-udrdr/rainy', real_image_dir="/mnt/d/PythonFiles/assignment/dataset/GT-RAIN_train",image_dir='rainy', label_dir='d rainy', num_samples=10,transform=transform)

# Create data loaders.
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Train and test the model
train_big_model(train_loader, num_epochs=1, device=device)

