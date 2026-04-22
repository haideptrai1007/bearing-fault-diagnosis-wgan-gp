import timm
import torch.nn as nn


def mobilenetv4(num_classes: int = 4, in_chans: int = 1, pretrained: bool = False) -> nn.Module:
    return timm.create_model(
        "mobilenetv4_conv_small_050",
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_path_rate=0.1,
    )