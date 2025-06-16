import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from medmnist import INFO
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

def get_dataset(dataset_name, split, download=True, transform=None):
    info = INFO[dataset_name]
    dataset_class = info['python_class']
    dataset = dataset_class(split=split, transform=transform, download=download)
    return dataset

def get_dataloaders(dataset_name='pathmnist', batch_size=64, download=True):
    transform = get_transforms()

    train_dataset = get_dataset(dataset_name, 'train', download, transform)
    val_dataset   = get_dataset(dataset_name, 'val', download, transform)
    test_dataset  = get_dataset(dataset_name, 'test', download, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
