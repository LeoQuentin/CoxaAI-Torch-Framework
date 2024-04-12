from coxaaitorch.models.model_registry import registermodel
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
import torch.nn as nn
from typing import Tuple


def _create_ResNet(
    version: str,
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    # ------------------ Model Configuration ------------------ #

    # instanciate a config object
    model_config = AutoConfig.from_pretrained(version)

    # raise an error if pretrained channel mismatch. Note, don't need to check size as it can vary
    if pretrained and model_config.num_channels != channels:
        raise ValueError(
            f"Number of input channels does not match the pretrained model. Use {config.num_channels} channels"  # noqa
        )

    elif channels:
        model_config.num_channels = channels

    # Add config arg to the config object
    if config:
        for key, value in config.items():
            setattr(model_config, key, value)

    if pretrained:
        model = AutoModelForImageClassification.from_pretrained(
            version, config=model_config
        )
    else:
        model = AutoModelForImageClassification.from_config(config=model_config)

    # Change the classifier layer to match the number of classes
    model.classifier = nn.Linear(model.classifier.in_features, classes)

    image_preprocessor = AutoImageProcessor.from_pretrained(version)
    if size:
        image_preprocessor.size = size

    return {
        "model": model,
        "processor": image_preprocessor,
        "model_ID": version,
        "source": "transformers",
    }


# ------------------ ResNet ------------------ #


@registermodel("ResNet-18")
def resnet_18(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ResNet(
        "microsoft/resnet-18",
        size,
        classes,
        pretrained,
        channels,
        config,
    )


@registermodel("ResNet-50")
def resnet_50(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ResNet(
        "microsoft/resnet-50",
        size,
        classes,
        pretrained,
        channels,
        config,
    )


@registermodel("ResNetV2-50x1-BiT")
def resnetv2_50x1_bit(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ResNet(
        "timm/resnetv2_50x1_bit.goog_in21k",
        size,
        classes,
        pretrained,
        channels,
        config,
    )
