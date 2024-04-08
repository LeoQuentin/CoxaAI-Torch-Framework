import torch  # noqa
import os
from datetime import timedelta
import dotenv
from coxaaitorch.augmentation.transforms import no_augmentation, random_augmentation

# for making the augmentation functions compatible with H5DataModule, hyperthreading/pickling issue
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

# from torch.optim.lr_scheduler import ReduceLROnPlateau

from coxaaitorch.utilities import H5DataModule
from coxaaitorch.models import BaseNetwork
from coxaaitorch.models import create_model

dotenv.load_dotenv()

project_root = os.getenv("PROJECT_ROOT")

log_dir = os.path.join(
    project_root, "coxaaitorch/experiments/SwinBinaryVsMulticlass/logs"
)

checkpoint_dir = os.path.join(
    project_root,
    "coxaaitorch/experiments/SwinBinaryVsMulticlass/checkpoints",
)


training_params = {
    "batch_size": 6,
    "early_stopping_patience": 25,
    "max_time_hours": 12,
    "train_folds": [0, 1, 2],
    "val_folds": [3],
    "test_folds": [4],
    "log_every_n_steps": 10,
    "presicion": "16-mixed",
    "lr_scheduler_factor": 0.2,
    "lr_scheduler_patience": 15,
    "learning_rate": 3e-4,
}


class NeuralNetwork(BaseNetwork):
    def __init__(self, model_name, num_classes, size, *args, **kwargs):
        self.model_dict = create_model(
            model_name, size=size, pretrained=True, classes=num_classes, channels=3
        )
        model = self.model_dict["model"]
        super().__init__(model, num_classes=num_classes, *args, **kwargs)
        self.learning_rate = 3e-4

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    for binary_or_multiclass in ["binary", "multiclass"]:
        model_name = "swinv2_base_patch4_window12to24_192to384_22kto1k_ft"
        num_classes = 2 if binary_or_multiclass == "binary" else 5
        size = (640, 640)

        # Create the model
        model = NeuralNetwork(model_name=model_name, num_classes=num_classes, size=size)

        preprocessor = model.model_dict["processor"]

        def train_transform(image):
            image = random_augmentation(image, size=size, channels=3)
            image = preprocessor(
                images=image,
                return_tensors="pt",
                input_data_format="channels_first",
                do_rescale=False,
            )
            image = image["pixel_values"]
            if len(image.size()) == 4:
                image = image.squeeze(0)
            return image

        def val_transform(image):
            image = no_augmentation(image, size=size, channels=3)
            image = preprocessor(
                images=image,
                return_tensors="pt",
                input_data_format="channels_first",
                do_rescale=False,
            )
            image = image["pixel_values"]
            if len(image.size()) == 4:
                image = image.squeeze(0)
            return image

        # Define the data module
        data_module = H5DataModule(
            data_file=os.getenv("DATA_FILE"),
            batch_size=training_params["batch_size"],
            train_folds=training_params["train_folds"],
            val_folds=training_params["val_folds"],
            test_folds=training_params["test_folds"],
            target_var="target" if binary_or_multiclass == "binary" else "diagnosis",
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=val_transform,
        )

        # Define the logger
        logger = CSVLogger(log_dir, name="swinV2" + "-" + binary_or_multiclass)

        # Define the callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=training_params["early_stopping_patience"]
        )
        model_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"swinV2-{binary_or_multiclass}" + "-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # Define the trainer
        trainer = Trainer(
            logger=logger,
            callbacks=[early_stopping, model_checkpoint, lr_monitor],
            max_time=timedelta(hours=training_params["max_time_hours"]),
            log_every_n_steps=training_params["log_every_n_steps"],
            precision=training_params["presicion"],
            accelerator="auto",
        )

        tuner = Tuner(trainer)
        tuner.lr_find(model, data_module, min_lr=1e-7, max_lr=3e-3, num_training=100)

        # Fit the model
        trainer.fit(model, data_module)

        # Test the model
        trainer.test(model, datamodule=data_module)
