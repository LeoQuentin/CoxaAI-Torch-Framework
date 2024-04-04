from coxaaitorch.models.model_registry import registermodel
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
import torch.nn as nn
from typing import Tuple


def _create_Swin(
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


# ------------------ Swin ------------------ #


# https://huggingface.co/microsoft/swin-base-patch4-window12-384-in22k
@registermodel("swin_base_patch4_window12_384_in22k")
def swin_base_patch4_window12_384_in22k(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_Swin(
        "microsoft/swin-base-patch4-window12-384-in22k",
        size,
        classes,
        pretrained,
        channels,
        config,
    )


# microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft
@registermodel("swinv2_base_patch4_window12to24_192to384_22kto1k_ft")
def swinv2_base_patch4_window12to24_192to384_22kto1k_ft(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_Swin(
        "microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft",
        size,
        classes,
        pretrained,
        channels,
        config,
    )
