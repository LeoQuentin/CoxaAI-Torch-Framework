from coxaaitorch.models.model_registry import registermodel
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
from typing import Tuple
import torch.nn as nn


def _create_efficientnet(
    version: str,
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    # ------------------ Model Configuration ------------------ #

    # instanciate a config object
    model_config = AutoConfig.from_pretrained(f"google/efficientnet-{version}")

    # raise an error if pretrained channel mismatch.
    if pretrained and model_config.num_channels != channels:
        raise ValueError(
            f"Number of input channels does not match the pretrained model. Use {model_config.num_channels} channels"  # noqa
        )

    elif channels:
        model_config.num_channels = channels

    if size:
        model_config.image_size = size

    # Add config arg to the config object
    if config:
        for key, value in config.items():
            setattr(model_config, key, value)

    if pretrained:
        model = AutoModelForImageClassification.from_pretrained(
            f"google/efficientnet-{version}", config=model_config
        )
    else:
        model = AutoModelForImageClassification.from_config(config=model_config)

    # Change the classifier layer to match the number of classes
    model.classifier = nn.Linear(model.classifier.in_features, classes)

    # ------------------ Image Processor ------------------ #

    image_preprocessor = AutoImageProcessor.from_pretrained(
        f"google/efficientnet-{version}"
    )
    if size:
        image_preprocessor.size = size

    return {
        "model": model,
        "processor": image_preprocessor,
        "model_ID": f"google/efficientnet_{version}",
        "source": "transformers",
    }


@registermodel("efficientnet_b0")
def efficientnet_b0(*args, **kwargs):
    return _create_efficientnet("b0", *args, **kwargs)


@registermodel("efficientnet_b1")
def efficientnet_b1(*args, **kwargs):
    return _create_efficientnet("b1", *args, **kwargs)


@registermodel("efficientnet_b2")
def efficientnet_b2(*args, **kwargs):
    return _create_efficientnet("b2", *args, **kwargs)


@registermodel("efficientnet_b3")
def efficientnet_b3(*args, **kwargs):
    return _create_efficientnet("b3", *args, **kwargs)


@registermodel("efficientnet_b4")
def efficientnet_b4(*args, **kwargs):
    return _create_efficientnet("b4", *args, **kwargs)


@registermodel("efficientnet_b5")
def efficientnet_b5(*args, **kwargs):
    return _create_efficientnet("b5", *args, **kwargs)


@registermodel("efficientnet_b6")
def efficientnet_b6(*args, **kwargs):
    return _create_efficientnet("b6", *args, **kwargs)


@registermodel("efficientnet_b7")
def efficientnet_b7(*args, **kwargs):
    return _create_efficientnet("b7", *args, **kwargs)
