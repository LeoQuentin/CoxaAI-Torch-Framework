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

from coxaaitorch.utilities import H5DataModule, print_experiment_metrics
from coxaaitorch.models import BaseNetwork, create_model

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
    "swin_base_patch4_window12_384_in22k",
    "ResNet-18",
    "ResNet-50"
]

training_params = {
    "batch_size": 8,
    "early_stopping_patience": 15,
    "max_time_hours": 12,
    "train_folds": [0, 1, 2],
    "val_folds": [3],
    "test_folds": [4],
    "log_every_n_steps": 10,
    "presicion": "16-mixed",
    "lr_scheduler_factor": 0.2,
    "lr_scheduler_patience": 8,
    "learning_rate": 3e-4,
}


class NeuralNetwork(BaseNetwork):
    def __init__(self, model_name, num_classes, size, *args, **kwargs):
        self.model_dict = create_model(
            model_name, size=size, pretrained=False, classes=num_classes, channels=3
        )
        model = self.model_dict["model"]
        super().__init__(model, num_classes=num_classes, *args, **kwargs)
        self.learning_rate = 3e-4

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
    logger_paths = []
    for model_name in models_to_train:
        for size in [(384, 384), (640, 640), (800, 800)]:
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

            model = NeuralNetwork(model_name=model_name, num_classes=2, size=size)

            # --------------------- DataModule ---------------------
            data_module = H5DataModule.from_base_config(
                        {
                            "train_transform": partial(
                                light_augmentation, size=size[0], channels=3
                            ),
                            "val_transform": partial(
                                no_augmentation, size=size[0], channels=3
                            ),
                            "test_transform": partial(
                                no_augmentation, size=size[0], channels=3
                            ),
                        }
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
            trainer.fit(model, data_module)

            trainer.test(model, datamodule=data_module)

            logger_paths.append(logger.log_dir)

    metrics = print_experiment_metrics(logger_paths)
    # open file and write
    with open(f"{log_dir}/metrics.txt", "w") as file:
        file.write(metrics)
