# utils.py
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def accuracy_topk(output, target, topk=(1,5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    return res  # list of tensors

def compute_classification_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            topk = accuracy_topk(logits, y, (1, 5))
            top1_correct += topk[0].item() * x.size(0) / 100
            top5_correct += topk[1].item() * x.size(0) / 100
            total += x.size(0)

            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    top1 = top1_correct / total * 100
    top5 = top5_correct / total * 100

    return {
        'top1': top1,
        'top5': top5,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'cm': cm
    }
