# model.py
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

def build_densenet121(num_classes:int=100, pretrained:bool=True):
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    in_f = model.classifier.in_features
    model.classifier = nn.Linear(in_f, num_classes)
    return model
