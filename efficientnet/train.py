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
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from itertools import product

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
    
    # Tối ưu CUDA
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    return device, True

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Áp dụng Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, loader, criterion, optimizer, device, scaler, accum_steps=1, use_mixup=False, mixup_alpha=1.0, epoch=1):
    model.train()
    total_loss = 0
    total_acc = 0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}")):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha, device=device)
        with autocast():
            logits = model(x)
            if use_mixup:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y)
            loss = loss / accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * x.size(0)
        acc = utils.accuracy_topk(logits, y, (1,))[0].item()
        total_acc += acc * x.size(0) / 100
        n += x.size(0)
    return total_loss / n, total_acc / n

def val_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_acc = 0
    n = 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Val Epoch {epoch}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_acc += utils.accuracy_topk(logits, y, (1,))[0].item() * x.size(0) / 100
            n += x.size(0)
    return total_loss / n, total_acc / n

def evaluate_metrics(model, loader, device):
    """Tính toán các metric bổ sung: precision, recall, F1-score"""
    metrics = utils.compute_classification_metrics(model, loader, device)
    return metrics

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
    all_log_path = f"efficientnetb0/logs/all_experiments_{tuning_hyperparameters}.csv"
    os.makedirs("efficientnetb0/logs", exist_ok=True)

    log_path = os.path.join("efficientnetb0/logs", f"{label}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "f1_score"])
        for i in range(len(stats['train_loss'])):
            writer.writerow([i+1, stats['train_loss'][i], stats['train_acc'][i], stats['val_loss'][i], stats['val_acc'][i], stats['f1_score'][i]])

    header_needed = not os.path.exists(all_log_path)
    with open(all_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["label", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "f1_score"])
        for i in range(len(stats['train_loss'])):
            writer.writerow([label, i+1, stats['train_loss'][i], stats['train_acc'][i], stats['val_loss'][i], stats['val_acc'][i], stats['f1_score'][i]])

def save_best_config(best_config, best_val_acc, tuning_hyperparameters):
    os.makedirs("efficientnetb0/logs", exist_ok=True)
    with open(f"efficientnetb0/logs/best_config_{tuning_hyperparameters}.json", "w") as f:
        json.dump({
            "best_config": best_config,
            "best_val_acc": best_val_acc
        }, f, indent=4)

def plot_all_results(tuning_hyperparameters):
    df = pd.read_csv(f"efficientnetb0/logs/all_experiments_{tuning_hyperparameters}.csv")
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
    plt.savefig(f"efficientnetb0/logs/all_loss_plot_{tuning_hyperparameters}.png")
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
    plt.savefig(f"efficientnetb0/logs/all_acc_plot_{tuning_hyperparameters}.png")
    plt.close()

def run_experiment(tuning_hyperparameters, config):
    print(f"Running config: {config['label']}")
    device, is_cuda = setup_cuda()
    
    # Load data từ dataset_loader gốc
    loaders = load_cifar100_datasets(
        data_dir="data/",
        batch_size=config['batch_size'],
        num_workers=4,
        augment=config['aug_type'],
        val_split=5000
    )
    
    # Thêm resize 224x224 và điều chỉnh RandomCrop, dùng chuẩn hóa ImageNet
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    def adjust_transform(subset, aug_type):
        original_dataset = subset.dataset
        original_transform = original_dataset.transform
        transforms_list = [transforms.Resize((224, 224))]
        for transform in original_transform.transforms:
            if isinstance(transform, transforms.RandomCrop) and transform.size[0] == 32:
                transforms_list.append(transforms.RandomCrop(224, padding=4))
            else:
                transforms_list.append(transform)
        return transforms.Compose(transforms_list)
    
    # Áp dụng transform mới cho train và val
    loaders['train'].dataset.dataset.transform = adjust_transform(loaders['train'].dataset, config['aug_type'])
    loaders['val'].dataset.dataset.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load EfficientNet-B0 và thêm dropout
    model_instance = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model_instance.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),  # EfficientNet-B0 đã có dropout, điều chỉnh nhẹ
        nn.Linear(model_instance.classifier[1].in_features, 100)
    )
    nn.init.xavier_uniform_(model_instance.classifier[1].weight)
    model_instance = model_instance.to(device)
    
    # Differential learning rate
    params = [
        {'params': [p for n, p in model_instance.named_parameters() if 'classifier' not in n], 'lr': config['lr'] * 0.1},
        {'params': model_instance.classifier.parameters(), 'lr': config['lr']}
    ]
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    else:  # adamw
        optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler()
    
    stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'f1_score': []}
    best_val_acc = 0.0
    early_stop_counter = 0
    patience = config.get('patience', 20)
    best_model_path = f"efficientnetb0/models/{config['label']}_best.pth"
    
    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        # Warm-up learning rate trong 5 epoch đầu
        if epoch <= 5:
            lr_scale = epoch / 5.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr'] * lr_scale * (1 if 'classifier' in param_group.get('name', '') else 0.1)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        use_mixup = False  # Tắt Mixup để kiểm tra hiệu quả
        tr_loss, tr_acc = train_epoch(
            model_instance,
            loaders['train'],
            criterion,
            optimizer,
            device,
            scaler,
            accum_steps=config['accum_steps'],
            use_mixup=use_mixup,
            mixup_alpha=1.0,
            epoch=epoch
        )
        val_loss, val_acc = val_epoch(model_instance, loaders['val'], criterion, device, epoch)
        metrics = evaluate_metrics(model_instance, loaders['val'], device)
        scheduler.step()

        print(f"[{config['label']}] Epoch {epoch}: "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
              f"F1: {metrics['f1']:.4f}")

        stats['train_loss'].append(tr_loss)
        stats['train_acc'].append(tr_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)
        stats['f1_score'].append(metrics['f1'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            os.makedirs("efficientnetb0/models", exist_ok=True)
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
            break

        plot_epoch_results(stats, epoch, config['label'], f"efficientnetb0/logs/{config['label']}")

        epoch_results = {
            'epoch': epoch,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'f1_score': metrics['f1'],
            'top1': metrics['top1'],
            'top5': metrics['top5'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }
        os.makedirs(f"efficientnetb0/logs/{config['label']}", exist_ok=True)
        with open(f"efficientnetb0/logs/{config['label']}/epoch_{epoch}_results.json", 'w') as f:
            json.dump(epoch_results, f, indent=4)

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱ Total training time for {config['label']}: {duration:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

    save_logs(stats, config['label'], tuning_hyperparameters)
    with open(f"efficientnetb0/logs/train_times_{tuning_hyperparameters}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(f"efficientnetb0/logs/train_times_{tuning_hyperparameters}.csv").st_size == 0:
            writer.writerow(["label", "duration_seconds"])
        writer.writerow([config['label'], round(duration, 2)])

    return best_val_acc

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    device, is_cuda = setup_cuda()
    
    param_grid = {
        "lr": [0.0005, 0.001, 0.01],
        "batch_size": [64, 128],
        "epochs": [30, 50],
        "weight_decay": [5e-4, 1e-3],
        "optimizer": ['sgd', 'adamw']
    }

    # Tạo tất cả tổ hợp tham số
    param_combinations = [
        dict(zip(param_grid.keys(), values))
        for values in product(*param_grid.values())
    ]

    tuning_hyperparameters = "lr_aug_report"
    best_val_acc_overall = 0.0
    best_config_overall = None

    for param in param_combinations:
        config = param.copy()
        config['label'] = (
            f"lr_{config['lr']}_bs_{config['batch_size']}_"
            f"epochs_{config['epochs']}_wd_{config['weight_decay']}_"
            f"opt_{config['optimizer']}_autoaug"
        )
        config['aug_type'] = 'aug_autoaugment'
        config['patience'] = 10
        config['accum_steps'] = 1

        best_val_acc = run_experiment(tuning_hyperparameters, config)
        
        if best_val_acc > best_val_acc_overall:
            best_val_acc_overall = best_val_acc
            best_config_overall = config.copy()

    plot_all_results(tuning_hyperparameters)
    
    # Lưu cấu hình tốt nhất
    if best_config_overall:
        save_best_config(best_config_overall, best_val_acc_overall, tuning_hyperparameters)
        print(f"Best configuration: {best_config_overall}")
        print(f"Best validation accuracy: {best_val_acc_overall*100:.2f}%")
    
    print("All experiments completed.")