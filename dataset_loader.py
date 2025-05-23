# dataset_loader.py

from typing import Optional, Dict
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# CIFAR-100 normalization constants
MEAN = (0.5071, 0.4865, 0.4409)
STD  = (0.2673, 0.2564, 0.2762)


def get_train_transform(aug_type="aug_none", normalize=None):
    if aug_type == "aug_none":
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == "aug_flip":
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == "aug_crop":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == "aug_flip_crop":
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == "aug_flip_crop_jitter":
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == "aug_flip_crop_erasing":
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing()
        ])

    elif aug_type == "aug_autoaugment":
        train_tf = transforms.Compose([
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == "aug_random_combo":
        train_tf = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                transforms.RandomErasing()
            ], p=0.8),
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == "aug_random_only":
        random_aug = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.RandomErasing(p=1.0)
        ])
        train_tf = transforms.Compose([
            random_aug,
            transforms.ToTensor(),
            normalize
        ])

    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

    return train_tf


def load_cifar100_datasets(
    data_dir: str,
    augment: str = "aug_none",
    val_split: Optional[int] = 5000,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Load CIFAR-100 train/val/test splits with optional basic augmentation,
    and wrap them in DataLoader objects.

    Args:
        data_dir: path to download/store CIFAR-100
        augment: if "aug_none", no augmentation; if "aug_basic", apply basic augmentation
        val_split: number of samples to hold out from train for validation;
                   if <=0 or None, no validation split
        seed: random seed for reproducible split
        batch_size: batch size for DataLoader
        num_workers: number of worker processes for data loading
        pin_memory: whether to pin memory in DataLoader

    Returns:
        A dict with keys 'train', 'val' (if val_split>0), and 'test',
        mỗi key mapping tới một torch.utils.data.DataLoader.
    """
    # 1) Chuẩn hóa – dùng chung cho val/test
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    common_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 2) Tạo biến train_tf với augmentation
    train_tf = get_train_transform(aug_type=augment, normalize=normalize)

    # 3) Tải dataset
    full_train = CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
    test_set   = CIFAR100(root=data_dir, train=False, download=True, transform=common_tf)

    # 4) Tách validation nếu cần
    datasets: Dict[str, torch.utils.data.Dataset] = {}
    if val_split and val_split > 0:
        train_n = len(full_train) - val_split
        train_set, val_set = random_split(
            full_train,
            [train_n, val_split],
            generator=torch.Generator().manual_seed(seed)
        )

        val_set.dataset.transform = common_tf
        datasets['train'] = train_set
        datasets['val']   = val_set
    else:
        datasets['train'] = full_train

    datasets['test'] = test_set

    # 5) Bọc thành DataLoader
    loaders: Dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    return loaders
