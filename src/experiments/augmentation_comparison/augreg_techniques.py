import torch
from torchvision import transforms
import os
import sys
import dotenv

dotenv.load_dotenv()
project_root = os.getenv("PROJECT_ROOT")
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal  # noqa
from src.utilities.H5DataModule import H5DataModule  # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL  # noqa
from src.augmentation.autoaugment import ImageNetPolicy  # noqa


def no_augreg(image, size=(384, 384)):
    # convert to PIL image as torchvision transforms work with PIL images
    image = np_image_to_PIL(image)

    # define augments
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)

    # Remove the batch dimension if it exists, as it sometimes gets added.
    if len(image.size()) == 4:
        image = image.squeeze(0)
    return image


def light_augreg(image, size=(384, 384)):
    image = np_image_to_PIL(image)

    # Preprocess the image
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)

    if len(image.size()) == 4:
        image = image.squeeze(0)
    return image


def heavy_augreg(image, resolution)