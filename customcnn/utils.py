#utils

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json

def accuracy_topk(logits, y, k=(1,)):
    with torch.no_grad():
        _, pred = logits.topk(max(k), 1, True, True)
        pred = pred.t()
        correct = pred.eq(y.view(1, -1).expand_as(pred))
        return [float(correct[:k_].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / logits.size(0) * 100 for k_ in k]

def compute_classification_metrics(model, testloader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct_top1, correct_top5, total = 0, 0, 0
    all_preds, all_labels = [], []
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, pred_top1 = torch.max(outputs, 1)
            _, pred_top5 = outputs.topk(5, dim=1)
            correct_top1 += (pred_top1 == labels).sum().item()
            correct_top5 += sum(labels[i].item() in pred_top5[i] for i in range(labels.size(0)))
            total += labels.size(0)
            
            all_preds.extend(pred_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total
    test_loss = test_loss / len(testloader)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'top1': top1_acc,
        'top5': top5_acc,
        'loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm.tolist(),
        'all_preds': all_preds,
        'all_labels': all_labels
    }
    
    return results

def visualize_misclassified(testloader, all_preds, all_labels, exp_dir, num_images=10, suffix=''):
    testset = testloader.dataset
    classes = testset.classes
    misclassified = [(img, pred, true) for img, (pred, true) in enumerate(zip(all_preds, all_labels)) if pred != true]
    
    plt.figure(figsize=(15, 5))
    for i, (idx, pred, true) in enumerate(misclassified[:num_images]):
        img, _ = testset[idx]
        img = img.permute(1, 2, 0).numpy() * np.array([0.2675, 0.2565, 0.2761]) + np.array([0.5071, 0.4867, 0.4408])
        img = np.clip(img, 0, 1)
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.title(f'Pred: {classes[pred]}\nTrue: {classes[true]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f'misclassified_images{suffix}.png'))
    plt.close()

def visualize_confusion_matrix(cm, classes, exp_dir, suffix=''):
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(exp_dir, f'confusion_matrix{suffix}.png'))
    plt.close()
