# eval.py
import torch
from .model import build_densenet121
from dataset_loader import load_cifar100_datasets
from utils import compute_classification_metrics

def main():
    device = torch.device("cuda")
    model = build_densenet121(pretrained=False).to(device)
    model.load_state_dict(torch.load("densenet121/models/lr_005.pth"))
    loaders = load_cifar100_datasets("data/", batch_size=128,
                              num_workers=4, augment=False, val_split=0)
    results = compute_classification_metrics(model, loaders['test'], device)
    print(f"Top-1 Acc: {results['top1']:.2f}%")
    print(f"Top-5 Acc: {results['top5']:.2f}%")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1']:.4f}")
    print("Confusion Matrix:")
    print(results['cm'])

if __name__ == "__main__":
    main()