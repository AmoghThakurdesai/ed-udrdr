import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(os.path.join(root_dir, image_dir)))
        self.label_files = [f for f in sorted(os.listdir(os.path.join(root_dir, label_dir))) if f.endswith('.jpg') or f.endswith('.png')]

        # Check if there are identically matching image label pairs in the directories
        # assert len(self.image_files) == len(self.label_files), \
        #     "The image and label directories do not contain identically matching image label pairs."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # TODO: Add some code to indicate that this is only for the rainy_dataset synthetic images repo

        
        image_file = self.image_files[idx].split('.')[0].split("_")[0] + '_' + str(random.randint(1, 14)) + '.jpg'
        
        image_path = os.path.join(self.root_dir, self.image_dir, image_file)
        
        label_path = os.path.join(self.root_dir, self.image_dir, self.image_files[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label