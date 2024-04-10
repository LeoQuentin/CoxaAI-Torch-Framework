import torch  # noqa
import os
from datetime import timedelta
import dotenv
from coxaaitorch.augmentation.transforms import (
    no_augmentation,
    random_augmentation,
    # autoaugment_policy_augmentation,
    # light_augmentation,
)
import matplotlib.pyplot as plt
from functools import partial

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
    project_root, "coxaaitorch/experiments/augmentation_comparison/swin/logs"
)

checkpoint_dir = os.path.join(
    project_root,
    "coxaaitorch/experiments/augmentation_comparison/swin/checkpoints",
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
        model_name = "swin_base_patch4_window12_384_in22k"
        num_classes = 2 if binary_or_multiclass == "binary" else 5
        size = (640, 640)

        # Create the model
        model = NeuralNetwork(model_name=model_name, num_classes=num_classes, size=size)

        preprocessor = model.model_dict["processor"]

        # Define the data module
        data_module = H5DataModule.from_base_config(
            {
                "train_transform": partial(
                    random_augmentation, size=640, channels=3, preprocessor=preprocessor
                ),
                "val_transform": partial(
                    no_augmentation, size=640, channels=3, preprocessor=preprocessor
                ),
                "test_transform": partial(
                    no_augmentation, size=640, channels=3, preprocessor=preprocessor
                ),
            }
        )

        # Define the logger
        logger = CSVLogger(log_dir, name="swin" + "-" + binary_or_multiclass)

        # Define the callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=training_params["early_stopping_patience"]
        )
        model_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"swin-{binary_or_multiclass}" + "-{epoch:02d}-{val_loss:.2f}",
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
        )

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            data_module,
            min_lr=1e-6,
            max_lr=3e-3,
            num_training=200,
            mode="linear",
        )
        print(lr_finder.results)

        # Save the resulting plot
        fig = lr_finder.plot(suggest=True)
        plt.savefig(f"{logger.log_dir}/lr_finder_plot.png")

        # Fit the model
        trainer.fit(model, data_module)

        # Test the model
        trainer.test(model, datamodule=data_module)
