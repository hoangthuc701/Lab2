import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes=100, pretrained=True):
    """
    Khởi tạo mô hình ResNet50 với lớp cuối được điều chỉnh cho CIFAR-100.
    
    Args:
        num_classes (int): Số lớp đầu ra (mặc định là 100 cho CIFAR-100).
        pretrained (bool): Có sử dụng trọng số huấn luyện trước từ ImageNet không.
    
    Returns:
        nn.Module: Mô hình ResNet50 đã điều chỉnh.
    """
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model