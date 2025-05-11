# train.py
import csv
import os
import torch, torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from .model import build_densenet121
from dataset_loader import load_cifar100_datasets
from utils import accuracy_topk
import matplotlib.pyplot as plt
import pandas as pd
import time

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0; total_acc = 0; n=0
    for x,y in tqdm(loader, desc="Train"):
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()*x.size(0)
        total_acc  += accuracy_topk(logits,y,(1,))[0].item()*x.size(0)/100
        n += x.size(0)
    return total_loss/n, total_acc/n

def save_logs(stats, label, tuning_hyperparameters):
    all_log_path = f"densenet121/logs/all_experiments_{tuning_hyperparameters}.csv"
    os.makedirs("densenet121/logs", exist_ok=True)

    log_path = os.path.join("densenet121/logs", f"{label}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc"])
        for i in range(len(stats['train_loss'])):
            writer.writerow([i+1, stats['train_loss'][i], stats['train_acc'][i]])

    header_needed = not os.path.exists(all_log_path)
    with open(all_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["label", "epoch", "train_loss", "train_acc"])
        for i in range(len(stats['train_loss'])):
            writer.writerow([label, i+1, stats['train_loss'][i], stats['train_acc'][i]])

def plot_all_results(tuning_hyperparameters):
    df = pd.read_csv(f"densenet121/logs/all_experiments_{tuning_hyperparameters}.csv")
    labels = df['label'].unique()

    plt.figure()
    for label in labels:
        sub = df[df['label'] == label]
        plt.plot(sub['epoch'], sub['train_loss'], label=label)
    plt.title("Train Loss vs Epochs")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid()
    plt.savefig(f"densenet121/logs/all_loss_plot_{tuning_hyperparameters}.png")

    plt.figure()
    for label in labels:
        sub = df[df['label'] == label]
        plt.plot(sub['epoch'], sub['train_acc'], label=label)
    plt.title("Train Accuracy vs Epochs")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.grid()
    plt.savefig(f"densenet121/logs/all_acc_plot_{tuning_hyperparameters}.png")

def run_experiment(tuning_hyperparameters, config):
    print(f"Running config: {config['label']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = load_cifar100_datasets("data/", batch_size=config['batch_size'], num_workers=2, augment=False, val_split=None)
    model = build_densenet121(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    stats = {'train_loss': [], 'train_acc': []}
    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, loaders['train'], criterion, optimizer, device)
        scheduler.step()

        print(f"[{config['label']}] Epoch {epoch}: "
              f"Train {tr_loss:.4f}/{tr_acc:.2f}% | ")
        
        stats['train_loss'].append(tr_loss)
        stats['train_acc'].append(tr_acc)

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱ Total training time for {config['label']}: {duration:.2f} seconds")

    # Save model checkpoint
    os.makedirs("densenet121/models", exist_ok=True)
    torch.save(model.state_dict(), f"densenet121/models/{config['label']}.pth")

    # Save training stats
    save_logs(stats, config['label'], tuning_hyperparameters)


if __name__=="__main__":

    # Turning learning rate
    tuning_hyperparameters = "lr"
    configs = [
        {'label': 'lr_01',  'lr': 0.1,  'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
        {'label': 'lr_005', 'lr': 0.05, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
        {'label': 'lr_001', 'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
    ]


    for config in configs:
        run_experiment(tuning_hyperparameters, config)
    plot_all_results(tuning_hyperparameters)
    print("All experiments completed.")

# This script trains a DenseNet-121 model on CIFAR-100 using different configurations of optimizers and learning rates.
# It includes functions for training, plotting results, and running experiments with different configurations.