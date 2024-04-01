import os
import sys
import dotenv
from torchvision import transforms


dotenv.load_dotenv()
project_root = os.getenv("PROJECT_ROOT")
if project_root:
    sys.path.append(project_root)
from src.utilities.np_image_to_PIL import np_image_to_PIL  # noqa


def train_augments(image, size):
    # image is a numpy array in the shape (H, W, C)
    image = np_image_to_PIL(image)  # convert to PIL image

    # Preprocess the image
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)


def val_test_augments(image, size):
    # basically same as train_preprocess but without the augmentations
    image = np_image_to_PIL(image)  # convert to PIL image

    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)
