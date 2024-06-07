from torch.utils.data import DataLoader
from torchvision import transforms
from data import CustomDataset
import torch
import torch.nn as nn
from torch.optim import Adam
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
print(transform)

test_dataset = CustomDataset(root_dir='/content/ed-udrdr/rainy', real_image_dir="/content/drive/MyDrive/Machine_Learning/Paper/Custom/GT-RAIN_train",image_dir='rainy', label_dir='d rainy', num_samples=10,transform=transform)
print(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print(test_loader)
def test_model(model_dict, test_loader, device):
    DN_state_dict = model_dict["DN_state_dict"]
    DN = RainEstimationNetwork()
    DN.load_state_dict(DN_state_dict)
    DN = DN.to(device)
    # Set the model to evaluation mode
    DN.eval()

    enc_state_dict = model_dict["enc_state_dict"]
    enc = Encoder()
    enc.load_state_dict(enc_state_dict)
    enc = enc.to(device)
    # Set the model to evaluation mode
    enc.eval()

    dec_state_dict = model_dict["dec_state_dict"]
    dec = Decoder()
    dec.load_state_dict(dec_state_dict)
    dec = dec.to(device)
    # Set the model to evaluation mode
    dec.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():  # No need to track gradients in testing
        for i, (x_syn, y_gt, _) in enumerate(test_loader):
            x_syn = x_syn.to(device)
            
            y_gt = y_gt.to(device)
            
            # Forward pass
            x_syn_enc = enc(x_syn)
            enc_outputs = DN(x_syn_enc)
            outputs = dec(enc_outputs)
            # Compute loss
            loss = criterion(outputs, y_gt)

            print(f'Test Step [{i+1}/{len(test_loader)}], Loss: {loss.item()}')
modeldict = torch.load("/content/drive/MyDrive/models_07062024_numsamples100_epoch1_Batchsize8.pth")
print(modeldict)
    

test_model(modeldict, test_loader, device)