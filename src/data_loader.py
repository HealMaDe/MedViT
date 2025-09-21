import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class MedMNISTDataset(Dataset):
    def __init__(self, split, dataset_name, npz_path, transform=None):
        data = np.load(npz_path)
        
        self.images, self.labels = data[f"{split}_images"], data[f"{split}_labels"]
        self.labels = self.labels.reshape(-1)  # flatten
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

        # handle grayscale (BreastMNIST, PneumoniaMNIST) vs RGB (DermaMNIST, RetinaMNIST)
        if img.ndim == 2:  # grayscale
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
        elif img.ndim == 3:  # RGB already
            img = Image.fromarray((img * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size):
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor()
    ])
    val_transform = T.Compose([T.ToTensor()])
    return train_transform, val_transform
