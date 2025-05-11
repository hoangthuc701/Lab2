# train.py
import torch, torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from .model import build_densenet121
from dataset_loader import load_cifar100_datasets
from utils import accuracy_topk

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
    total_loss=0; total_acc=0; n=0
    with torch.no_grad():
      for x,y in tqdm(loader, desc="Val"):
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()*x.size(0)
        total_acc  += accuracy_topk(logits,y,(1,))[0].item()*x.size(0)/100
        n += x.size(0)
    return total_loss/n, total_acc/n

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load CIFAR-100 dataset
    loaders = load_cifar100_datasets("data/", batch_size=128, num_workers=4,
                              augment=True, val_split=5000)
    model = build_densenet121(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc = 0
    for epoch in range(1,101):
        tr_loss, tr_acc = train_epoch(model, loaders['train'],
                                      criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, loaders['val'],
                                      criterion, device)
        scheduler.step()
        print(f"[Epoch {epoch:03d}] "
              f"Train: {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_densenet121.pth")

if __name__=="__main__":
    main()
