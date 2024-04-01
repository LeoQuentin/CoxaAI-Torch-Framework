import torch
from torchvision import models
from transformers import AutoConfig, AutoModelForImageClassification


def efficientnet_v2_s(resolution, num_classes):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    return model


def efficientnet_v2_m(resolution, num_classes):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    return model


def efficientnet_v2_l(resolution, num_classes):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    return model


def efficientnet(version, resolution, num_classes):
    model_id = f"google/efficientnet-{version}"
    config = AutoConfig.from_pretrained(model_id)
    config.image_size = resolution
    model = AutoModelForImageClassification.from_config(config)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model


def get_model(model_name, resolution, num_classes):
    if model_name == "efficientnet_v2_s":
        return efficientnet_v2_s(resolution, num_classes)
    elif model_name == "efficientnet_v2_m":
        return efficientnet_v2_m(resolution, num_classes)
    elif model_name == "efficientnet_v2_l":
        return efficientnet_v2_l(resolution, num_classes)
    elif model_name in [
        f"efficientnet-{version}"
        for version in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    ]:
        version = model_name.split("-")[1]
        return efficientnet(version, resolution, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
