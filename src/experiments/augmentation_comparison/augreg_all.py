import torch
from torchvision import transforms, models
import dotenv
import os
import sys
from model_definitions import get_model

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau


dotenv.load_dotenv()
project_root = os.getenv("PROJECT_ROOT")
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal  # noqa
from src.utilities.H5DataModule import H5DataModule  # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL  # noqa
from src.augmentation.autoaugment import ImageNetPolicy  # noqa


if __name__ == "__main__":
    # Model ID
    for model_name in [
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
    ]:
        # Training parameters
        training_params = {
            "model_id": model_name,
            "batch_size": (
                32
                if model_name
                in [
                    "efficientnet-b0",
                    "efficientnet-b1",
                    "efficientnet-b2",
                    "efficientnet-b3",
                ]
                else 16
            ),
            "early_stopping_patience": 12,
            "max_time_hours": 12,
            "train_folds": [0, 1, 2],
            "val_folds": [3],
            "test_folds": [4],
            "log_every_n_steps": 10,
            "presicion": "16-mixed",
            "size": 384,
            "lr_scheduler_factor": 0.2,
            "lr_scheduler_patience": 5,
        }

        # --------------------- Model ---------------------

        class NeuralNetwork(BaseNormalAbnormal):
            def __init__(self, *args, **kwargs):
                # Initialize the EfficientNetV2 model
                model = get_model(model_name, 384, 2)
                super().__init__(model=model, *args, **kwargs)

                # set learning rate
                self.learning_rate = 3e-4

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                lr_scheduler = {
                    "scheduler": ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=training_params["lr_scheduler_factor"],  # noqa
                        patience=training_params["lr_scheduler_patience"],
                    ),  # noqa
                    "monitor": "val_loss",  # Specify the metric you want to monitor
                    "interval": "epoch",
                    "frequency": 1,
                }
                return [optimizer], [lr_scheduler]
