#train.py

import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
import json

# Add directories to sys.path for Kaggle
sys.path.append(os.getcwd())  # Current working directory (/kaggle/working/)
# If files are in a dataset, uncomment and replace 'your-dataset' with the actual dataset name
# sys.path.append('/kaggle/input/your-dataset/')
print("Python module search paths:", sys.path)  # Debug: Show sys.path

import model
import dataset_loader
import utils

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    n = 0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += utils.accuracy_topk(logits, y, (1,))[0] * x.size(0) / 100
        n += x.size(0)
    return total_loss / n, total_acc / n

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    n = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Clear GPU memory
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Val")):
            print(f"Validation batch {i+1}")  # Debug: Log batch number
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_acc += utils.accuracy_topk(logits, y, (1,))[0] * x.size(0) / 100
            n += x.size(0)
    return total_loss / n, total_acc / n

def plot_epoch_results(stats, epoch, label, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, epoch + 1))
    plt.plot(epochs, stats['train_loss'][:epoch], label='Train Loss')
    plt.plot(epochs, stats['val_loss'][:epoch], label='Val Loss')
    plt.title(f'Loss vs Epochs (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f'epoch_{epoch}_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, stats['train_acc'][:epoch], label='Train Accuracy')
    plt.plot(epochs, stats['val_acc'][:epoch], label='Val Accuracy')
    plt.title(f'Accuracy vs Epochs (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f'epoch_{epoch}_acc.png'))
    plt.close()

def save_logs(stats, label, tuning_hyperparameters):
    all_log_path = f"convnext/logs/all_experiments_{tuning_hyperparameters}.csv"
    os.makedirs("convnext/logs", exist_ok=True)

    log_path = os.path.join("convnext/logs", f"{label}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(stats['train_loss'])):
            writer.writerow([i+1, stats['train_loss'][i], stats['train_acc'][i], stats['val_loss'][i], stats['val_acc'][i]])

    header_needed = not os.path.exists(all_log_path)
    with open(all_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["label", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(stats['train_loss'])):
            writer.writerow([label, i+1, stats['train_loss'][i], stats['train_acc'][i], stats['val_loss'][i], stats['val_acc'][i]])

def plot_all_results(tuning_hyperparameters):
    df = pd.read_csv(f"convnext/logs/all_experiments_{tuning_hyperparameters}.csv")
    labels = df['label'].unique()

    plt.figure(figsize=(10, 5))
    for label in labels:
        sub = df[df['label'] == label]
        plt.plot(sub['epoch'], sub['train_loss'], label=f'{label} Train')
        plt.plot(sub['epoch'], sub['val_loss'], label=f'{label} Val', linestyle='--')
    plt.title("Train and Val Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"convnext/logs/all_loss_plot_{tuning_hyperparameters}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for label in labels:
        sub = df[df['label'] == label]
        plt.plot(sub['epoch'], sub['train_acc'], label=f'{label} Train')
        plt.plot(sub['epoch'], sub['val_acc'], label=f'{label} Val', linestyle='--')
    plt.title("Train and Val Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"convnext/logs/all_acc_plot_{tuning_hyperparameters}.png")
    plt.close()

def run_experiment(tuning_hyperparameters, config):
    print(f"Running config: {config['label']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = dataset_loader.load_cifar100_datasets("data/", batch_size=config['batch_size'], num_workers=0, augment=True, val_split=5000)
    model_instance = model.build_convnext(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model_instance, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model_instance, loaders['val'], criterion, device)
        scheduler.step()

        print(f"[{config['label']}] Epoch {epoch}: "
              f"Train {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Val {val_loss:.4f}/{val_acc:.2f}%")

        stats['train_loss'].append(tr_loss)
        stats['train_acc'].append(tr_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)

        plot_epoch_results(stats, epoch, config['label'], f"convnext/logs/{config['label']}")

        epoch_results = {
            'epoch': epoch,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        os.makedirs(f"convnext/logs/{config['label']}", exist_ok=True)
        with open(f"convnext/logs/{config['label']}/epoch_{epoch}_results.json", 'w') as f:
            json.dump(epoch_results, f, indent=4)

        os.makedirs("convnext/models", exist_ok=True)
        torch.save(model_instance.state_dict(), f"convnext/models/{config['label']}_epoch_{epoch}.pth")

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱ Total training time for {config['label']}: {duration:.2f} seconds")

    save_logs(stats, config['label'], tuning_hyperparameters)
    with open(f"convnext/logs/train_times_{tuning_hyperparameters}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(f"convnext/logs/train_times_{tuning_hyperparameters}.csv").st_size == 0:
            writer.writerow(["label", "duration_seconds"])
        writer.writerow([config['label'], round(duration, 2)])

if __name__ == "__main__":
    # For Kaggle notebook compatibility, ensure correct sys.path
    sys.path.append(os.getcwd())
    # If files are in a dataset, uncomment and replace 'your-dataset' with the actual dataset name
    # sys.path.append('/kaggle/input/your-dataset/')
    print("Python module search paths:", sys.path)  # Debug: Show sys.path
    tuning_hyperparameters = "lr"
    configs = [
        {'label': 'lr_001', 'lr': 0.001, 'batch_size': 4, 'optimizer': 'adam', 'weight_decay': 1e-4, 'epochs': 30},
        {'label': 'lr_0005', 'lr': 0.0005, 'batch_size': 4, 'optimizer': 'adam', 'weight_decay': 1e-4, 'epochs': 30},
        {'label': 'lr_0001', 'lr': 0.0001, 'batch_size': 4, 'optimizer': 'adam', 'weight_decay': 1e-4, 'epochs': 30},
    ]

    for config in configs:
        run_experiment(tuning_hyperparameters, config)
    plot_all_results(tuning_hyperparameters)
    print("All experiments completed.")
