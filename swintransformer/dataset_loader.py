#dataset_loader.py

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_cifar100_datasets(data_dir, batch_size=4, num_workers=0, augment=True, val_split=0):
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10.),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]) if augment else transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    loaders = {'train': trainloader, 'test': testloader}
    
    if val_split > 0:
        from torch.utils.data import Subset
        import numpy as np
        indices = np.arange(len(trainset))
        np.random.shuffle(indices)
        train_indices = indices[val_split:]
        val_indices = indices[:val_split]
        print(f"Train subset size: {len(train_indices)}, Val subset size: {len(val_indices)}")  # Debug
        train_subset = Subset(trainset, train_indices)
        val_subset = Subset(trainset, val_indices)
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)  # num_workers=0 for stability
        loaders['train'] = trainloader
        loaders['val'] = valloader
    
    return loaders
