# dataset_loader.py

from typing import Optional, Dict
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import random_split

# CIFAR-100 normalization constants
MEAN = (0.5071, 0.4865, 0.4409)
STD  = (0.2673, 0.2564, 0.2762)

def load_cifar100_datasets(
    data_dir: str,
    augment: bool = True,
    val_split: Optional[int] = 5000,
    seed: int = 42
) -> Dict[str, torch.utils.data.Dataset]:
    """
    Load CIFAR-100 train/val/test splits, applying exactly one random augmentation per image when augment=True.

    Args:
        data_dir: path to download/store CIFAR-100
        augment: if True, wrap each train image in a RandomChoice of augmentations
        val_split: number of samples to hold out from train for validation;
                   if None or 0, no validation split is performed
        seed: random seed for reproducible split

    Returns:
        A dict with keys 'train', 'val' (if val_split), and 'test',
        each mapping to a torch.utils.data.Dataset.
    """
    # Normalize transform
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    common_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        # Exactly one of these augmentations will be applied per image
        augmentation = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        ])
        train_transforms = transforms.Compose([
            augmentation,
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transforms = common_transforms

    # Download datasets
    full_train = CIFAR100(root=data_dir, train=True,  download=True, transform=train_transforms)
    test_set   = CIFAR100(root=data_dir, train=False, download=True, transform=common_transforms)

    # Create validation split if requested
    if val_split and val_split > 0:
        train_size = len(full_train) - val_split
        train_set, val_set = random_split(
            full_train,
            [train_size, val_split],
            generator=torch.Generator().manual_seed(seed)
        )
        # override transform for validation
        val_set.dataset.transform = common_transforms
        return {
            'train': train_set,
            'val'  : val_set,
            'test' : test_set
        }

    return {
        'train': full_train,
        'test' : test_set
    }
