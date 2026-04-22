import timm
import torch.nn as nn


def tinynet_d(num_classes: int = 4, in_chans: int = 1, pretrained: bool = False) -> nn.Module:
    return timm.create_model(
        "tinynet_d",
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_path_rate=0.2,
    )