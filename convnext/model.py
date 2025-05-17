#model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x

class CNBlock(nn.Module):
    def __init__(self, dim, layer_scale: float, stochastic_depth_prob: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            nn.LayerNorm((dim,), eps=1e-6),  # Applied after permute
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block[0](x)  # Conv2d
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.block[1](x)  # LayerNorm
        x = self.block[2](x)  # Linear
        x = self.block[3](x)  # GELU
        x = self.block[4](x)  # Linear
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.layer_scale * x
        x = self.stochastic_depth(x)
        x = x + identity
        return x

class StochasticDepth(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device)).to(x.dtype)
        return x * mask / keep_prob

class CNBlockConfig:
    def __init__(self, input_channels: int, out_channels: int, num_layers: int):
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: list,
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 100
    ):
        super().__init__()
        layers = []
        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, firstconv_output_channels, kernel_size=4, stride=4, padding=0, bias=True),
                LayerNorm2d(firstconv_output_channels)
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            stage = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0) if total_stage_blocks > 1 else 0.0
                stage.append(CNBlock(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2)
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        lastblock = block_setting[-1]
        lastconv_output_channels = lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        self.classifier = nn.Sequential(
            LayerNorm2d(lastconv_output_channels),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channels, num_classes)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def build_convnext(num_classes=100):
    # ConvNeXt-Tiny configuration for CIFAR-100
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    return ConvNeXt(block_setting, stochastic_depth_prob=0.1, num_classes=num_classes)
