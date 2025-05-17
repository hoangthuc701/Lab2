# extract_features.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
import numpy as np
from .model import build_feature_extractor

def extract_and_save_features(split="train", batch_size=64, save_path="features_train.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = CIFAR100(root='./data', train=(split=="train"), download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_feature_extractor(pretrained=True).to(device).eval()

    features, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        feats = model(imgs)  # [B, 512]
        features.append(feats.cpu().numpy())
        labels.append(lbls.numpy())

    X = np.concatenate(features)
    y = np.concatenate(labels)
    np.savez(save_path, X=X, y=y)
    print(f"Saved features to {save_path}")

if __name__ == "__main__":
    extract_and_save_features(split="train", save_path="svm/features_train.npz")
    extract_and_save_features(split="test", save_path="svm/features_test.npz")
