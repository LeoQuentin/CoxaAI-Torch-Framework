import torch # noqa
from torchvision import transforms
import os
from datetime import timedelta
import sys
import dotenv

# huggingface model
from transformers import AutoConfig, AutoModelForImageClassification
# Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal # noqa
from src.utilities.H5DataModule import H5DataModule # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL # noqa
from src.augmentation.autoaugment import ImageNetPolicy # noqa

size = (384, 384)

def train_preprocess(image):
    # image is a numpy array in the shape (H, W, C)
    image = np_image_to_PIL(image)  # convert to PIL image

    # Preprocess the image
    transform_pipeline = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomRotation(10),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    image = transform_pipeline(image)

    # Remove the batch dimension if it exists
    if len(image.size()) == 4:
        image = image.squeeze(0)
    return image


def val_test_preprocess(image):
    # basically same as train_preprocess but without the augmentations
    image = np_image_to_PIL(image)  # convert to PIL image

    transform_pipeline = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    image = transform_pipeline(image)

    if len(image.size()) == 4:
        image = image.squeeze(0)
    return image


class EfficientNet_384(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModelForImageClassification.from_config(config)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 2)

    def forward(self, x):
        return self.model(x)


checkpoint_path = "/mnt/users/leobakh/VET_project/VET-Special-syllabus/src/experiments/initial_model_comparison/binary/lightAugReg/modelcheckpoints/google/efficientnet-b0_binary_lightAugReg_384_best_checkpoint_epoch=23_val_loss=0.25.ckpt"  # noqa
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint["state_dict"]

model_id = "google/efficientnet-b0"  # Replace with the appropriate model ID
config = AutoConfig.from_pretrained(model_id)
config.image_size = (384, 384)  # Replace with the appropriate image size
config.num_channels = 1

model = EfficientNet_384(config)
model.load_state_dict(state_dict)
