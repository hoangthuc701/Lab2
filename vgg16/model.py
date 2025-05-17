# model.py
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

def build_vgg16(num_classes: int = 100, pretrained: bool = True):
    # Load pretrained weights nếu cần
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)
    
    # Sửa phần classifier để phù hợp với số lớp mới (100 lớp)
    in_f = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_f, num_classes)
    
    return model
