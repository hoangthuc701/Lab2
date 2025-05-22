# model_vit.py
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit_b16(num_classes: int = 100, pretrained: bool = True):
    """
    Builds a Vision Transformer (ViT-B/16) model and adjusts the final classification head.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        nn.Module: A ViT-B/16 model with a modified classifier.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    
    # Replace the head for fine-tuning
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model
