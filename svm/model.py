# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet18(weights=weights)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # remove FC layer

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)              # [B, 512, 1, 1]
            x = x.view(x.size(0), -1)         # Flatten to [B, 512]
        return x

def build_feature_extractor(pretrained=True):
    return ResNetFeatureExtractor(pretrained=pretrained)
