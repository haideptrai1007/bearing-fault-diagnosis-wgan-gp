import timm
import torch.nn as nn


def edgenext_xxs(num_classes: int = 4, in_chans: int = 1, pretrained: bool = False) -> nn.Module:
    return timm.create_model(
        "edgenext_xx_small.in1k",
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )