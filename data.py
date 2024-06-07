import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, real_image_dir, num_samples, transform=None,consider_real=True):
        self.consider_real=consider_real
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.real_image_dir = real_image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(os.path.join(root_dir, image_dir)))
        self.real_image_files = []
        for dirpath, dirnames, filenames in os.walk(self.real_image_dir):
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    relative_path = os.path.relpath(os.path.join(dirpath, filename), self.real_image_dir)
                    self.real_image_files.append(relative_path)

        self.label_files = [f for f in sorted(os.listdir(os.path.join(root_dir, label_dir))) if f.endswith('.jpg') or f.endswith('.png')]
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly select an image pair from the synthetic dataset
        synthetic_idx = random.randint(0, len(self.image_files) - 1)
        print(f"Synthetic: {synthetic_idx}")
        if(self.image_dir == "/content/rainy-image-dataset/rainy image"):
          image_file = self.image_files[synthetic_idx].split('.')[0].split("_")[0] + '_' + str(random.randint(1, 14)) + '.jpg'
        else:
          image_file = self.image_files[synthetic_idx]
        # the above image file construction only works when rainy_image_dataset is considered.
        # it doesnt work for any generalised dataset 
        print(f"image file: {image_file}")
        
        image_path = os.path.join(self.root_dir, self.image_dir, image_file)
        print(f"image_path: {image_path}")
        
        
        label_path = os.path.join(self.root_dir, self.image_dir, self.image_files[synthetic_idx])
        print(f"label_path: {label_path}")
        
        
        # Load and transform the images
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        if self.consider_real:
        # Randomly select a real rainy image
          real_idx = random.randint(0, len(self.real_image_files) - 1)
          real_image_file = self.real_image_files[real_idx]
          real_image_path = os.path.join(self.real_image_dir, real_image_file)
          real_image = Image.open(real_image_path).convert('RGB')

          if self.transform:
            real_image = self.transform(real_image)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        print(real_image_path, image_path, label_path)

        return image, label, real_image

