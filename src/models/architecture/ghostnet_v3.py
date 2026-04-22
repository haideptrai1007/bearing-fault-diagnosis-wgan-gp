import timm
import torch.nn as nn


def ghostnetv3(num_classes: int = 4, in_chans: int = 1, pretrained: bool = False) -> nn.Module:
    return timm.create_model(
        "ghostnetv3_050",
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
    )
