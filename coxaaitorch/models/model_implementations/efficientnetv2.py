from coxaaitorch.models.model_registry import registermodel
from torchvision import models
import torch.nn as nn
from typing import Tuple


def _create_efficientnetv2(
    version: str,
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = None,
    config: dict = None,
):
    # raise an error if pretrained channel or res mismatch.
    if pretrained and (channels != 3 or size != (384, 384)):
        raise ValueError(
            "Pretrained model is only available for input size of 384x384 and 3 channels"
        )

    # Instanciate with pretrained weights
    if pretrained:
        if version == "s":
            model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        elif version == "m":
            model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
        elif version == "l":
            model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported model: {version}")

    # Instanciate without pretrained weights
    else:
        if version == "s":
            model = models.efficientnet_v2_s(weights=None)
        elif version == "m":
            model = models.efficientnet_v2_m(weights=None)
        elif version == "l":
            model = models.efficientnet_v2_l(weights=None)
        else:
            raise ValueError(f"Unsupported model: {version}")

    # Change the classifier layer to match the number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)

    # Image Processor (None)
    return {
        "model": model,
        "processor": None,  # No image processor, as opposed to f.ex ViT and Swin
        "model_ID": f"efficientnet_v2_{version}",
        "source": "torchvision",
    }


# https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
@registermodel("efficientnet_v2_s")
def efficientnet_v2_s(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_efficientnetv2("s", size, classes, pretrained, channels, config)


# https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_m.html#torchvision.models.efficientnet_v2_m
@registermodel("efficientnet_v2_m")
def efficientnet_v2_m(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_efficientnetv2("m", size, classes, pretrained, channels, config)


# https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
@registermodel("efficientnet_v2_l")
def efficientnet_v2_l(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_efficientnetv2("l", size, classes, pretrained, channels, config)
