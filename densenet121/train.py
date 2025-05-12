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

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0; total_acc = 0; n = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_acc += accuracy_topk(logits, y, (1,))[0].item() * x.size(0) / 100
            n += x.size(0)
    return total_loss / n, total_acc / n

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
    # Load data
    df = pd.read_csv(f"densenet121/logs/all_experiments_{tuning_hyperparameters}.csv")
    labels = df['label'].unique()

    # Load early stopping info
    stop_df = pd.read_csv(f"densenet121/logs/train_times_{tuning_hyperparameters}.csv")

    # --- Plot Train Loss ---
    plt.figure(figsize=(10, 5))
    for label in labels:
        sub = df[df['label'] == label]
        stop_epoch = stop_df[stop_df['label'] == label]['early_stop_epoch'].values[0]
        y_val = sub[sub['epoch'] == stop_epoch]['train_loss'].values[0]
        plt.plot(sub['epoch'], sub['train_loss'], label=label)
        plt.scatter([stop_epoch], [y_val], color='red', marker='x', zorder=5)
    plt.title("Train Loss vs Epochs")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(f"densenet121/logs/all_loss_plot_{tuning_hyperparameters}.png")

    # --- Plot Train Accuracy ---
    plt.figure(figsize=(10, 5))
    for label in labels:
        sub = df[df['label'] == label]
        stop_epoch = stop_df[stop_df['label'] == label]['early_stop_epoch'].values[0]
        y_val = sub[sub['epoch'] == stop_epoch]['train_acc'].values[0]
        plt.plot(sub['epoch'], sub['train_acc'], label=label)
        plt.scatter([stop_epoch], [y_val], color='red', marker='x', zorder=5)
    plt.title("Train Accuracy vs Epochs")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(f"densenet121/logs/all_acc_plot_{tuning_hyperparameters}.png")

def run_experiment(tuning_hyperparameters, config):
    print(f"Running config: {config['label']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = load_cifar100_datasets("data/", batch_size=config['batch_size'], num_workers=2, augment=False, val_split=5000)
    model = build_densenet121(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    opt = config['optimizer'].lower()

    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.0, weight_decay=config['weight_decay'])  # mặc định không momentum
    elif opt == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    # Early stopping config
    patience = 10
    best_val_loss = float("inf")
    trigger_times = 0

    for epoch in range(1, config['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, loaders['val'], criterion, device)
        scheduler.step()

        print(f"[{config['label']}] Epoch {epoch}: "
              f"Train {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Val {val_loss:.4f}/{val_acc:.2f}%")

        stats['train_loss'].append(tr_loss)
        stats['train_acc'].append(tr_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
             # Save model checkpoint
            os.makedirs("densenet121/models", exist_ok=True)
            torch.save(model.state_dict(), f"densenet121/models/{config['label']}.pth")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"⛔ Early stopping triggered at epoch {epoch}")
                break

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱ Total training time for {config['label']}: {duration:.2f} seconds")

    # Save training stats
    save_logs(stats, config['label'], tuning_hyperparameters)
    with open(f"densenet121/logs/train_times_{tuning_hyperparameters}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(f"densenet121/logs/train_times_{tuning_hyperparameters}.csv").st_size == 0:
            writer.writerow(["label", "duration_seconds", "early_stop_epoch"])
        writer.writerow([config['label'], round(duration, 2), epoch])


if __name__=="__main__":

    # Turning learning rate
    # tuning_hyperparameters = "lr"
    # configs = [
    #     {'label': 'lr_01',  'lr': 0.1,  'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
    #     {'label': 'lr_005', 'lr': 0.05, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
    #     {'label': 'lr_001', 'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
    # ]

    # Turning batch size
    # tuning_hyperparameters = "bs"
    # configs = [
    #     {'label': 'bs_128','lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
    #     {'label': 'bs_64', 'lr': 0.01, 'batch_size': 64, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
    #     {'label': 'bs_32', 'lr': 0.01, 'batch_size': 32, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
    # ]


    # Turning optimizers
    # tuning_hyperparameters = "opt"
    # configs = [
    #     {'label': 'opt_sgd',      'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd',      'weight_decay': 1e-4, 'epochs': 50},
    #     {'label': 'opt_momentum', 'lr': 0.01, 'batch_size': 128, 'optimizer': 'momentum', 'weight_decay': 1e-4, 'epochs': 50},
    #     {'label': 'opt_adam',     'lr': 0.01,'batch_size': 128, 'optimizer': 'adam',     'weight_decay': 1e-4, 'epochs': 50},
    #     {'label': 'opt_adamw',    'lr': 0.01,'batch_size': 128, 'optimizer': 'adamw',    'weight_decay': 1e-4, 'epochs': 50},
    #     {'label': 'opt_rmsprop',  'lr': 0.01, 'batch_size': 128, 'optimizer': 'rmsprop',  'weight_decay': 1e-4, 'epochs': 50},
    # ]

    # Turning weight decay
    tuning_hyperparameters = "wd"
    configs = [
        {'label': 'wd_0',   'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 0.0,  'epochs': 50},
        {'label': 'wd_5e4','lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 5e-4, 'epochs': 50},
        {'label': 'wd_1e4','lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50},
        {'label': 'wd_1e3','lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-3, 'epochs': 50},

    ]

    for config in configs:
        run_experiment(tuning_hyperparameters, config)
    plot_all_results(tuning_hyperparameters)
    print("All experiments completed.")

# This script trains a DenseNet-121 model on CIFAR-100 using different configurations of optimizers and learning rates.
# It includes functions for training, plotting results, and running experiments with different configurations.