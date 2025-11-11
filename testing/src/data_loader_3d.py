import numpy as np
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MedMNIST3DDataset(Dataset):
    def __init__(self, npz_path, split="train", transform=None):
        data = np.load(npz_path)
        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"].squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (28, 28, 28)
        label = int(self.labels[idx])

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 28, 28, 28)

        img = img.repeat(3, 1, 1, 1)  # (3, 28, 28, 28)

        if self.transform:
            img = self.transform(img)



        return img, label


# 3D augmentation transforms
class RandomFlip3D:
    """Random flip for 3D data"""
    def __call__(self, x):
        # x shape: (C, D, H, W) for single image
        if x.dim() == 4:  # Single image: (C, D, H, W)
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])  # Flip height (dim 2)
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[3])  # Flip width (dim 3)
        return x

class RandomRotate3D:
    """Simple 3D rotation by swapping axes"""
    def __call__(self, x):
        if x.dim() == 4:  # Single image: (C, D, H, W)
            # Randomly choose rotation type
            rotation_type = torch.randint(0, 4, (1,)).item()
            if rotation_type == 1:
                # Rotate 90 degrees
                x = x.transpose(2, 3).flip(2)
            elif rotation_type == 2:
                # Rotate 180 degrees
                x = x.flip(2).flip(3)
            elif rotation_type == 3:
                # Rotate 270 degrees
                x = x.transpose(2, 3).flip(3)
        return x


def get_loaders(dataset_name,batch_size):
    npz_path = f"/media/massoud/New Volume/MedViT-main/ViT_3D/data/{dataset_name}.npz"

    # Enhanced transforms with working augmentation
    train_transform = transforms.Compose([
        RandomFlip3D(),
        RandomRotate3D(),
    ])

    val_transform = transforms.Compose([])

    train_ds = MedMNIST3DDataset(npz_path, split="train", transform=train_transform)
    val_ds = MedMNIST3DDataset(npz_path, split="val", transform=val_transform)
    test_ds = MedMNIST3DDataset(npz_path, split="test", transform=val_transform)

    num_classes = len(np.unique(train_ds.labels))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes
