#eval

import os
import sys
import torch
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd

# Add current working directory to sys.path for Kaggle
sys.path.append(os.getcwd())
# If files are in a dataset, uncomment and replace 'your-dataset' with the actual dataset name
# sys.path.append('/kaggle/input/your-dataset/')
print("Python module search paths:", sys.path)  # Debug: Show sys.path

import model
import dataset_loader
import utils

def evaluate_all_models(tuning_hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "custom_cnn/models"
    results_dir = "custom_cnn/results"
    os.makedirs(results_dir, exist_ok=True)

    loaders = dataset_loader.load_cifar100_datasets("data/", batch_size=4, num_workers=2, augment=False, val_split=0)
    test_loader = loaders['test']

    comparison_results = []
    train_times = pd.read_csv(f"custom_cnn/logs/train_times_{tuning_hyperparameters}.csv").set_index('label')

    for file in os.listdir(model_dir):
        if tuning_hyperparameters in file and file.endswith(".pth"):
            label = file.replace(".pth", "")
            model_path = os.path.join(model_dir, file)
            print(f"\n🔍 Evaluating: {label}")

            model_instance = model.build_custom_cnn(num_classes=100).to(device)
            model_instance.load_state_dict(torch.load(model_path))
            model_instance.eval()

            results = utils.compute_classification_metrics(model_instance, test_loader, device)

            print(f"Top-1 Accuracy: {results['top1']:.2f}%")
            print(f"Top-5 Accuracy: {results['top5']:.2f}%")
            print(f"Precision:      {results['precision']:.4f}")
            print(f"Recall:         {results['recall']:.4f}")
            print(f"F1-score:       {results['f1']:.4f}")
            print(f"Confusion Matrix:\n{results['cm']}")

            utils.visualize_misclassified(test_loader, results['all_preds'], results['all_labels'], results_dir, suffix=f'_{label}')
            utils.visualize_confusion_matrix(results['cm'], test_loader.dataset.classes, results_dir, suffix=f'_{label}')
            
            with open(os.path.join(results_dir, f'results_{label}.json'), 'w') as f:
                json.dump({
                    'label': label,
                    'top1': results['top1'],
                    'top5': results['top5'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1': results['f1'],
                    'cm': results['cm']
                }, f, indent=4)

            duration = train_times.loc[label.split('_epoch')[0]]['duration_seconds'] if label.split('_epoch')[0] in train_times.index else 0
            comparison_results.append({
                'label': label,
                'top1': results['top1'],
                'top5': results['top5'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'duration': duration
            })

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(results_dir, f'comparison_{tuning_hyperparameters}.csv'), index=False)

    plt.figure(figsize=(12, 6))
    labels = comparison_df['label']
    top1 = comparison_df['top1']
    top5 = comparison_df['top5']
    x = range(len(labels))
    plt.bar([i - 0.2 for i in x], top1, width=0.4, label='Top-1 Accuracy')
    plt.bar([i + 0.2 for i in x], top5, width=0.4, label='Top-5 Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Top-1 and Top-5 Accuracy')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'comparison_{tuning_hyperparameters}_plot.png'))
    plt.close()

def main():
    tuning_hyperparameters = "lr"
    evaluate_all_models(tuning_hyperparameters)

if __name__ == "__main__":
    # For Kaggle notebook compatibility, ensure correct sys.path
    sys.path.append(os.getcwd())
    # If files are in a dataset, uncomment and replace 'your-dataset' with the actual dataset name
    # sys.path.append('/kaggle/input/your-dataset/')
    print("Python module search paths:", sys.path)  # Debug: Show sys.path
    main()
