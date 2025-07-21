import os
from typing import List, Optional, Sequence, Union
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import torch
import medmnist
from medmnist import INFO

class RepeatChannels:
    """Repeat channels for consistent input shape, pickle-safe."""
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def __call__(self, tensor):
        return tensor.repeat(self.num_channels, 1, 1)

def debug_collate(batch):
    xs, ys = zip(*batch)
    # ✅ Replace labels with consistent dummy zeros
    dummy_labels = [torch.tensor(0) for _ in ys]
    return torch.utils.data.default_collate(list(zip(xs, dummy_labels)))

class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_names: List[str],
        data_dir: str = "Data/",
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        patch_size: Union[int, Sequence[int]] = 28,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_names = data_names
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.n_channels = max(INFO[name]['n_channels'] for name in self.data_names)

    def setup(self, stage: Optional[str] = None) -> None:
        os.makedirs(self.data_dir, exist_ok=True)

        train_datasets = []
        val_datasets = []

        for name in self.data_names:
            info = INFO[name]
            DataClass = getattr(medmnist, info['python_class'])
            n_channels = info['n_channels']

            transforms_list = [
                transforms.ToTensor(),
                transforms.Resize((self.patch_size, self.patch_size)),
            ]

            if n_channels < self.n_channels:
                transforms_list.insert(1, RepeatChannels(self.n_channels))

            transforms_list.append(transforms.Normalize(
                mean=[.5]*self.n_channels, std=[.5]*self.n_channels))

            transform_ = transforms.Compose(transforms_list)

            train_ds = DataClass(root=self.data_dir, split='train', transform=transform_, download=True)
            val_ds = DataClass(root=self.data_dir, split='test', transform=transform_, download=True)

            # Verify shapes match
            for i in range(3):
                xi, _ = train_ds[i]
                assert xi.shape == (self.n_channels, self.patch_size, self.patch_size), \
                    f"❌ MISMATCH: got {xi.shape}, expected ({self.n_channels},{self.patch_size},{self.patch_size})"

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

        print(f"✅ Combined Train Dataset: {len(self.train_dataset)}")
        print(f"✅ Combined Val Dataset: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            collate_fn=debug_collate  # ✅ clean collate: drops labels safely
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            collate_fn=debug_collate  # ✅ clean collate
        )

    def test_dataloader(self) -> DataLoader:

        return self.val_dataloader()
