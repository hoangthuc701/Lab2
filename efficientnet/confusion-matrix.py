import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend Agg để tránh lỗi Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import models, transforms
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
import pandas as pd

# Thêm parent directory vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_loader import load_cifar100_datasets
import utils

# Thiết lập CUDA và tối ưu cho RTX 4090
def setup_cuda():
    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu"), False
    device = torch.device("cuda")
    print(f"Using CUDA {torch.version.cuda} on device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    return device, True

# Dự đoán trên tập validation
def predict(model, loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Predicting"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
            preds = torch.argmax(logits, dim=1)
            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    return np.array(true_labels), np.array(pred_labels)

# Vẽ confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for CIFAR-100')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Thiết lập CUDA
    device, is_cuda = setup_cuda()

    # Cấu hình mô hình tốt nhất
    config = {
        'label': 'lr_0.0005_bs_64_epochs_50_wd_0.0005_opt_adamw_autoaug',
        'batch_size': 64,
        'aug_type': 'aug_autoaugment'
    }
    checkpoint_path = f"efficientnetb0/models/lr_0005_epoch_30.pth"

    # Tải mô hình
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),  # EfficientNet-B0 đã có dropout, điều chỉnh nhẹ
        nn.Linear(model.classifier[1].in_features, 100)
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    # Tải tập validation
    loaders = load_cifar100_datasets(
        data_dir="data/",
        batch_size=config['batch_size'],
        num_workers=2,
        augment=config['aug_type'],
        val_split=5000
    )
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    loaders['val'].dataset.dataset.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Dự đoán
    true_labels, pred_labels = predict(model, loaders['val'], device)

    # Danh sách 100 lớp của CIFAR-100 (theo thứ tự)
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    # Vẽ và lưu confusion matrix
    save_path = f"efficientnetb0/logs/{config['label']}_confusion_matrix.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_confusion_matrix(true_labels, pred_labels, class_names, save_path)
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    main()
