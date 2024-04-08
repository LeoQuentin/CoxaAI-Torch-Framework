import torch  # noqa
import os
from datetime import timedelta
import dotenv
from coxaaitorch.augmentation.transforms import no_augmentation, light_augmentation

# for making the augmentation functions compatible with H5DataModule, hyperthreading/pickling issue
from functools import partial


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from coxaaitorch.utilities import H5DataModule
from coxaaitorch.models import BaseNormalAbnormal
from coxaaitorch.models import create_model

dotenv.load_dotenv()

project_root = os.getenv("PROJECT_ROOT")
log_dir = os.path.join(
    project_root, "coxaaitorch/experiments/initial_comparison/outputs"
)  # noqa

checkpoint = os.path.join(
    project_root,
    "coxaaitorch/experiments/initial_comparison/checkpoints",
)  # noqa

models_to_train = [
    "vit-base-patch16-384",
    "swinv2_base_patch4_window12to24_192to384_22kto1k_ft",
    "swin_base_patch4_window12_384_in22k",
]

training_params = {
    "batch_size": 8,
    "early_stopping_patience": 20,
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


class NeuralNetwork(BaseNormalAbnormal):
    def __init__(self, model_name, size, training_params, *args, **kwargs):
        model = create_model(
            model_name, size=size, pretrained=False, classes=2, channels=1
        )
        super().__init__(model=model["model"], *args, **kwargs)
        self.training_params = training_params

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.training_params["learning_rate"]
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=training_params["lr_scheduler_factor"],
                patience=training_params["lr_scheduler_patience"],
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    for model_name in models_to_train:
        for size in [(384, 384)]:
            # --------------------- Parameters ---------------------
            training_params["size"] = size
            training_params["model_name"] = model_name
            if size == (384, 384):
                batch_size = 10
            elif size == (640, 640):
                batch_size = 8
            elif size == (800, 800):
                batch_size = 5
            training_params["batch_size"] = batch_size

            model = NeuralNetwork(model_name, size, training_params)

            # --------------------- DataModule ---------------------

            dm = H5DataModule(
                os.getenv("DATA_FILE"),
                batch_size=training_params["batch_size"],
                train_folds=training_params["train_folds"],
                val_folds=training_params["val_folds"],
                test_folds=training_params["test_folds"],
                target_var="target",
                train_transform=partial(light_augmentation, size=size, channels=1),
                val_transform=partial(no_augmentation, size=size, channels=1),
                test_transform=partial(no_augmentation, size=size, channels=1),
            )

            # --------------------- Train ---------------------

            # callbacks
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=training_params["early_stopping_patience"]
            )
            model_checkpoint = ModelCheckpoint(
                dirpath=checkpoint,
                filename=f"{model_name}_{size[0]}_best_checkpoint"
                + "_{epoch:02d}_{val_loss:.2f}",  # noqa
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )

            # Logger
            logger = CSVLogger(
                save_dir=log_dir,
                name=f"{model_name}_{size[0]}",
                flush_logs_every_n_steps=training_params["log_every_n_steps"],
            )

            # Trainer
            trainer = Trainer(
                max_time=timedelta(hours=training_params["max_time_hours"]),
                accelerator="auto",
                callbacks=[
                    early_stopping,
                    model_checkpoint,
                    LearningRateMonitor(logging_interval="step"),
                ],
                logger=logger,
                log_every_n_steps=training_params["log_every_n_steps"],
                precision=training_params["presicion"],
            )
            # Training
            trainer.fit(model, dm)

            trainer.test(model, datamodule=dm)
