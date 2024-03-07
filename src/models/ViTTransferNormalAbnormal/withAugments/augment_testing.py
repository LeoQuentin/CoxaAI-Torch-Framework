import torch # noqa
from torchvision import transforms
from PIL import Image
import numpy as np

# huggingface model
from transformers import ViTImageProcessor, ViTForImageClassification
# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datetime import timedelta
import os
import sys
import dotenv
dotenv.load_dotenv()

project_root = os.getenv('PROJECT_ROOT')
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal # noqa
from src.utilities.H5DataModule import H5DataModule # noqa
from src.utilities.AutoAugment.autoaugment import ImageNetPolicy


feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-384")


def preprocess_image(image):
    # image is a numpy array in the shape (H, W, C)
    image = (image * 255).astype(np.uint8)

    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)

    # Now convert to a PIL Image
    try:
        image = Image.fromarray(image)
    except TypeError as e:
        print(f"Error converting array to image: {e}")
        # Additional debugging info
        print(f"Array shape: {image.shape}, Array dtype: {image.dtype}")
        raise

    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor()
    ])
    image = transform_pipeline(image)

    data = feature_extractor(images=image,
                             return_tensors="pt",
                             input_data_format="channels_first",
                             do_rescale=False)
    data = {"pixel_values": image}
    pixel_values = data["pixel_values"]
    if pixel_values.shape[0] == 1:  # Check if the batch dimension is 1
        pixel_values = pixel_values.squeeze(0)  # Remove the first dimension
    return pixel_values


def val_test_preprocess(image):
    # image is a numpy array in the shape (H, W, C)
    image = (image * 255).astype(np.uint8)

    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)

    # Now convert to a PIL Image
    try:
        image = Image.fromarray(image)
    except TypeError as e:
        print(f"Error converting array to image: {e}")
        # Additional debugging info
        print(f"Array shape: {image.shape}, Array dtype: {image.dtype}")
        raise

    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    image = transform_pipeline(image)

    data = feature_extractor(images=image,
                             return_tensors="pt",
                             input_data_format="channels_first",
                             do_rescale=False)
    data = {"pixel_values": image}
    pixel_values = data["pixel_values"]
    if pixel_values.shape[0] == 1:  # Check if the batch dimension is 1
        pixel_values = pixel_values.squeeze(0)  # Remove the first dimension
    return pixel_values

# --------------------- DataModule ---------------------


dm = H5DataModule(os.getenv("DATA_FILE"),
                  batch_size=1,
                  train_folds=[0, 1, 2],
                  val_folds=[3],
                  test_folds=[4],
                  target_var='target',
                  train_transform=preprocess_image,
                  val_transform=val_test_preprocess,
                  test_transform=val_test_preprocess
                  )


# Run through a couple of batches to check if the preprocessing is working

if __name__ == '__main__':
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch[0].shape)
        break
