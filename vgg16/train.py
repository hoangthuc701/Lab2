# train.py (for VGG16)
import csv
import os
import torch, torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from .model import build_vgg16  # Đã sửa lại
from dataset_loader import load_cifar100_datasets
from utils import accuracy_topk
import matplotlib.pyplot as plt
import pandas as pd
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

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
    all_log_path = f"vgg16/logs/all_experiments_{tuning_hyperparameters}.csv"
    os.makedirs("vgg16/logs", exist_ok=True)

    log_path = os.path.join("vgg16/logs", f"{label}.csv")
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
    df = pd.read_csv(f"vgg16/logs/all_experiments_{tuning_hyperparameters}.csv")
    labels = df['label'].unique()
    stop_df = pd.read_csv(f"vgg16/logs/train_times_{tuning_hyperparameters}.csv")

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
    plt.savefig(f"vgg16/logs/all_loss_plot_{tuning_hyperparameters}.png")

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
    plt.savefig(f"vgg16/logs/all_acc_plot_{tuning_hyperparameters}.png")

def run_experiment(tuning_hyperparameters, config):
    print(f"Running config: {config['label']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = load_cifar100_datasets("data/", batch_size=config['batch_size'], num_workers=2, augment=config['aug_type'], val_split=5000)
    model = build_vgg16(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    opt = config['optimizer'].lower()

    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.0, weight_decay=config['weight_decay'])
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

    if config['scheduler'] == 'sched_cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'sched_step':
        scheduler = StepLR(optimizer, step_size=config.get('step', 30), gamma=config.get('gamma', 0.1))
    elif config['scheduler'] == 'sched_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    patience = 10
    best_val_loss = float("inf")
    trigger_times = 0

    for epoch in range(1, config['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, loaders['val'], criterion, device)
        
        if config['scheduler'] == 'sched_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"[{config['label']}] Epoch {epoch}: "
              f"Train {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Val {val_loss:.4f}/{val_acc:.2f}%")

        stats['train_loss'].append(tr_loss)
        stats['train_acc'].append(tr_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            os.makedirs("vgg16/models", exist_ok=True)
            torch.save(model.state_dict(), f"vgg16/models/{config['label']}.pth")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"⛔ Early stopping triggered at epoch {epoch}")
                break

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱ Total training time for {config['label']}: {duration:.2f} seconds")

    save_logs(stats, config['label'], tuning_hyperparameters)
    with open(f"vgg16/logs/train_times_{tuning_hyperparameters}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(f"vgg16/logs/train_times_{tuning_hyperparameters}.csv").st_size == 0:
            writer.writerow(["label", "duration_seconds", "early_stop_epoch"])
        writer.writerow([config['label'], round(duration, 2), epoch])


if __name__ == "__main__":

    # Turning learning rate
    # tuning_hyperparameters = "lr"
    # configs = [
    #     {'label': 'lr_01',  'lr': 0.1,  'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'lr_005', 'lr': 0.05, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'lr_001', 'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    # ]

    # Turning batch size
    # tuning_hyperparameters = "bs"
    # configs = [
    #     {'label': 'bs_128','lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'bs_64', 'lr': 0.01, 'batch_size': 64, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'bs_32', 'lr': 0.01, 'batch_size': 32, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    # ]

    # Turning optimizers
    # tuning_hyperparameters = "opt"
    # configs = [
    #     {'label': 'opt_sgd',      'lr': 0.05, 'batch_size': 64, 'optimizer': 'sgd',      'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'opt_momentum', 'lr': 0.01, 'batch_size': 64, 'optimizer': 'momentum', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'opt_adam',     'lr': 1e-3,'batch_size': 64, 'optimizer': 'adam',     'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'opt_adamw',    'lr': 5e-4,'batch_size': 64, 'optimizer': 'adamw',    'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'opt_rmsprop',  'lr': 1e-3, 'batch_size': 64, 'optimizer': 'rmsprop',  'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    # ]

    # Turning weight decay
    # tuning_hyperparameters = "wd"
    # configs = [
    #     {'label': 'wd_0',   'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 0.0,  'epochs': 50,  'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'wd_5e4','lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 5e-4, 'epochs': 50,  'aug_type': 'aug_none','scheduler': 'sched_cosine' },
    #     {'label': 'wd_1e4','lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50,  'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    #     {'label': 'wd_1e3','lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-3, 'epochs': 50,  'aug_type': 'aug_none','scheduler': 'sched_cosine'},
    # ]

    # Turning scheduler
    # tuning_hyperparameters = "sched"
    # configs = [
    #     {'label': 'sched_cosine',   'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'scheduler': 'sched_cosine', 'aug_type': 'aug_none'},
    #     {'label': 'sched_step',     'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'scheduler': 'sched_step', 'aug_type': 'aug_none'},
    #     {'label': 'sched_plateau',  'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'scheduler': 'sched_plateau', 'aug_type': 'aug_none'},
    # ]


    tuning_hyperparameters = "aug"
    configs = [
        {'label': 'aug_none',         'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_none','scheduler': 'sched_cosine'},
        {'label': 'aug_flip',         'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_flip', 'scheduler': 'sched_cosine'},
        {'label': 'aug_crop',         'lr': 0.01, 'batch_size':  128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_crop', 'scheduler': 'sched_cosine'},
        {'label': 'aug_flip_crop',    'lr': 0.01, 'batch_size':  128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_flip_crop', 'scheduler': 'sched_cosine'},
        {'label': 'aug_flip_crop_jitter',    'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_flip_crop_jitter', 'scheduler': 'sched_cosine'},
        {'label': 'aug_flip_crop_erasing',   'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_flip_crop_erasing', 'scheduler': 'sched_cosine'},
        {'label': 'aug_autoaugment',         'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': 'aug_autoaugment', 'scheduler': 'sched_cosine'},
        {'label': 'aug_random_combo',        'lr': 0.01, 'batch_size': 128, 'optimizer': 'sgd', 'weight_decay': 1e-4, 'epochs': 50, 'aug_type': "aug_random_combo", 'scheduler': 'sched_cosine'},
        {'label': "aug_random_only",         "lr": 0.01, "batch_size": 128, "optimizer": "sgd", "weight_decay": 1e-4, "epochs": 50, "aug_type": "aug_random_only", "scheduler": "sched_cosine"},
    ]

    for config in configs:
        run_experiment(tuning_hyperparameters, config)
    plot_all_results(tuning_hyperparameters)
    print("All experiments completed.")
