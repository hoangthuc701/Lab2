import os
import torch
from .model import build_vgg16  # Đổi sang hàm cho VGG16
from dataset_loader import load_cifar100_datasets
from utils import compute_classification_metrics

def evaluate_all_models(tuning_hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "vgg16/models"  # Đổi folder model

    # Load test set
    loaders = load_cifar100_datasets("data/", batch_size=128, num_workers=4, augment="aug_none", val_split=0)
    test_loader = loaders['test']

    # Loop through all model files with keyword
    for file in os.listdir(model_dir):
        if tuning_hyperparameters in file and file.endswith(".pth"):
            label = file.replace(".pth", "")
            model_path = os.path.join(model_dir, file)
            print(f"\n🔍 Evaluating: {label}")

            # Load model
            model = build_vgg16(pretrained=False).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Evaluate
            results = compute_classification_metrics(model, test_loader, device)

            # Print results
            print(f"Top-1 Accuracy: {results['top1']:.2f}%")
            print(f"Top-5 Accuracy: {results['top5']:.2f}%")
            print(f"Precision:      {results['precision']:.4f}")
            print(f"Recall:         {results['recall']:.4f}")
            print(f"F1-score:       {results['f1']:.4f}")
            print(f"Confusion Matrix:\n{results['cm']}")

def main():
    tuning_hyperparameters = "sched"
    evaluate_all_models(tuning_hyperparameters)

if __name__ == "__main__":
    main()
