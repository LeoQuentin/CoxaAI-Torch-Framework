from coxaaitorch.models.model_registry import registermodel
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
import torch.nn as nn
from typing import Tuple


def _create_ViT(
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

    if size and model_config.image_size != size:
        model_config.image_size = size
        monkey_patch = True

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

    if monkey_patch:
        # Overwrite the __call__ method to include the interpolate_pos_encoding argument
        def new_call(self, pixels, *args, **kwargs):
            return super(type(model), model).__call__(
                pixels, interpolate_pos_encoding=True, *args, **kwargs
            )

        # Monkey patch the __call__ method. Careful, this is a bit hacky!
        model.__call__ = new_call.__get__(model, type(model))

    # ------------------ Image Processor ------------------ #

    image_preprocessor = AutoImageProcessor.from_pretrained(version)
    if size:
        image_preprocessor.size = size

    return {
        "model": model,
        "processor": image_preprocessor,
        "model_ID": version,
        "source": "transformers",
    }


# ---------------------- Base ViT models ---------------------- #


# https://huggingface.co/google/vit-base-patch16-224
@registermodel("vit-base-patch16-224")
def ViTBasePatch16_224(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ViT(
        "google/vit-base-patch16-224", size, classes, pretrained, channels, config
    )


# https://huggingface.co/google/vit-base-patch16-384
@registermodel("vit-base-patch16-384")
def ViTBasePatch16_384(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ViT(
        "google/vit-base-patch16-384", size, classes, pretrained, channels, config
    )


# ---------------------- Large ViT models ---------------------- #


# https://huggingface.co/google/vit-large-patch16-224
@registermodel("vit-large-patch16-224")
def ViTLargePatch16_224(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ViT(
        "google/vit-large-patch16-224", size, classes, pretrained, channels, config
    )


# https://huggingface.co/google/vit-large-patch16-384
@registermodel("vit-large-patch16-384")
def ViTLargePatch16_384(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ViT(
        "google/vit-large-patch16-384", size, classes, pretrained, channels, config
    )


# ---------------------- Patch32 Base ViT models ---------------------- #


# https://huggingface.co/google/vit-base-patch32-224-in21k
@registermodel("vit-base-patch32-224-in21k")
def ViTBasePatch32_224_in21k(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ViT(
        "google/vit-base-patch32-224-in21k", size, classes, pretrained, channels, config
    )


# https://huggingface.co/google/vit-base-patch32-384
@registermodel("vit-base-patch32-384")
def ViTBasePatch32_384(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ViT(
        "google/vit-base-patch32-384", size, classes, pretrained, channels, config
    )


# ---------------------- Patch32 Large ViT models ---------------------- #


# https://huggingface.co/google/vit-large-patch32-224-in21k
@registermodel("vit-large-patch32-224-in21k")
def ViTLargePatch32_224_in21k(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ViT(
        "google/vit-large-patch32-224-in21k",
        size,
        classes,
        pretrained,
        channels,
        config,
    )


# https://huggingface.co/google/vit-large-patch32-384
@registermodel("vit-large-patch32-384")
def ViTLargePatch32_384(
    size: Tuple[int, int],
    classes: int = 2,
    pretrained: bool = False,
    channels: int = 3,
    config: dict = None,
):
    return _create_ViT(
        "google/vit-large-patch32-384", size, classes, pretrained, channels, config
    )
