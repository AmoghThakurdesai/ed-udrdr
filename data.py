import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, real_image_dir, num_samples, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.real_image_dir = real_image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(os.path.join(root_dir, image_dir)))
        self.real_image_files = sorted(os.listdir(real_image_dir))
        self.label_files = [f for f in sorted(os.listdir(os.path.join(root_dir, label_dir))) if f.endswith('.jpg') or f.endswith('.png')]
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly select an image pair from the synthetic dataset
        synthetic_idx = random.randint(0, len(self.image_files) - 1)
        image_file = self.image_files[synthetic_idx].split('.')[0].split("_")[0] + '_' + str(random.randint(1, 14)) + '.jpg'
        image_path = os.path.join(self.root_dir, self.image_dir, image_file)
        label_path = os.path.join(self.root_dir, self.image_dir, self.image_files[synthetic_idx])

        # Randomly select a real rainy image
        real_idx = random.randint(0, len(self.real_image_files) - 1)
        real_image_file = self.real_image_files[real_idx]
        real_image_path = os.path.join(self.real_image_dir, real_image_file)

        # Load and transform the images
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        real_image = Image.open(real_image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            real_image = self.transform(real_image)

        return image, label, real_image

