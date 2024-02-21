import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class HipsDataset(Dataset):
    def __init__(self, file_path, fold, transform=None):
        self.file_path = file_path
        self.fold = fold
        self.transform = transform

        with h5py.File(self.file_path, 'r') as file:
            self.images = file[f'{fold}/image'][:]
            self.targets = file[f'{fold}/target'][:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
