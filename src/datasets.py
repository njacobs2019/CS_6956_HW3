from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class ConditionalMNIST(Dataset):
    def __init__(self, root='./mnist-data', train=True, transform=None, download=True):
        if download and Path(root).exists():
            print(f"{root} already exists. Setting download to False.")
            download = False
        self.mnist = MNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        # Convert label to one-hot encoding
        label_onehot = torch.zeros(10)
        label_onehot[label] = 1.0
        return image, label_onehot, label

def get_dataloaders(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ConditionalMNIST(train=True, transform=transform, download=True)
    test_dataset = ConditionalMNIST(train=False, transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

# run it once for downloading the dataset
if __name__ == "__main__":
    get_dataloaders()
    print("Datasets downloaded and ready to use.")

