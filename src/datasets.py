from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

DS_ARRAYS = tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]


np.random.seed(0)


def get_synth_data_arrays() -> DS_ARRAYS:
    n = 10000

    mean1 = np.array([1.0, 5.0])
    sd1 = np.array([0.7, 0.3])

    mean2 = np.array([7.0, 3.0])
    sd2 = np.array([0.4, 0.3])

    data1 = np.hstack(
        (
            np.random.normal(loc=mean1, scale=sd1, size=(n // 2, 2)),
            np.zeros((n // 2, 1)),
        )
    )
    data2 = np.hstack(
        (np.random.normal(loc=mean2, scale=sd2, size=(n // 2, 2)), np.ones((n // 2, 1)))
    )

    data = np.vstack((data1, data2))
    data = data.astype(np.float32)
    np.random.shuffle(data)

    n_train = int(n * 0.8)

    train = data[:n_train]
    test = data[n_train:]

    X_train = train[:, :2]
    y_train = train[:, -1].reshape(-1, 1)

    X_test = test[:, :2]
    y_test = test[:, -1].reshape(-1, 1)

    return (X_train, y_train), (X_test, y_test)


def get_synth_ds(ds_arrays: DS_ARRAYS) -> tuple[Dataset, Dataset]:
    (X_train, y_train), (X_test, y_test) = ds_arrays

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    return train_ds, test_ds


class ConditionalMNIST(Dataset):
    def __init__(
        self,
        root: str = "./mnist-data",
        train: bool = True,
        transform: Callable | None = None,
        download: bool = True,
    ) -> None:
        if download and Path(root).exists():
            print(f"{root} already exists. Setting download to False.")
            download = False
        self.mnist = MNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Any]:
        image, label = self.mnist[idx]
        # Convert label to one-hot encoding
        label_onehot = torch.zeros(10)
        label_onehot[label] = 1.0
        return image, label_onehot, label


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    dataset_name: Literal["mnist", "synthetic"] = "mnist",
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    if dataset_name == "mnist":
        train_dataset = ConditionalMNIST(train=True, transform=transform, download=True)
        test_dataset = ConditionalMNIST(train=False, transform=transform, download=True)
    elif dataset_name == "synthetic":
        synth_data = get_synth_data_arrays()
        train_dataset, test_dataset = get_synth_ds(synth_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# run it once for downloading the dataset
if __name__ == "__main__":
    get_dataloaders()
    print("Datasets downloaded and ready to use.")
