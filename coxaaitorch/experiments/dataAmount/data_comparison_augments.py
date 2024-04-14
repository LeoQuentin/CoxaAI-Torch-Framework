from coxaaitorch.experiments.dataAmount.new_folds_datamodule import NewFoldsDataModule
from coxaaitorch.utilities import H5FoldDataset, print_experiment_metrics
import dotenv
import os
from coxaaitorch.models import BaseNetwork, create_model
import torch
from coxaaitorch.augmentation.transforms import no_augmentation, random_augmentation
from functools import partial
from datetime import timedelta

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

dotenv.load_dotenv()

project_root = os.getenv("PROJECT_ROOT")

log_dir = os.path.join(project_root, "coxaaitorch/experiments/dataAmount/logs")

checkpoint_dir = os.path.join(
    project_root,
    "coxaaitorch/experiments/dataAmount/checkpoints",
)


class NeuralNetwork(BaseNetwork):
    def __init__(self, model_name, num_classes, size, *args, **kwargs):
        self.model_dict = create_model(
            model_name, size=size, pretrained=False, classes=num_classes, channels=1
        )
        model = self.model_dict["model"]
        super().__init__(model, num_classes=num_classes, *args, **kwargs)
        self.learning_rate = 3e-4

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


initial_dataset = H5FoldDataset(
    file_path=os.getenv("DATA_FILE"),
    folds=[0, 1, 2],
    target_var="diagnosis",
    transform=partial(random_augmentation, size=384, channels=1, num_ops=2, magnitude=5),
)

val_dataset = H5FoldDataset(
    file_path=os.getenv("DATA_FILE"),
    folds=[3],
    transform=partial(no_augmentation, size=384, channels=1),
)

test_dataset = H5FoldDataset(
    file_path=os.getenv("DATA_FILE"),
    folds=[4],
    transform=partial(no_augmentation, size=384, channels=1),
)

if __name__ == "__main__":
    logger_directories = []
    for num_folds in range(1, 20):
        model_name = "efficientnet_b3"
        size = (384, 384)

        # Create the model
        model = NeuralNetwork(model_name=model_name, num_classes=2, size=size)

        datamodule = NewFoldsDataModule(
            dataset=initial_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=64,
            dataset_splits=20,
            used_folds=num_folds,
            target_var="target",
            train_loader_workers=16,
            val_loader_workers=16,
            test_loader_workers=6,
        )
        datamodule.prepare_data()
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()

        name = f"efficientnet-b3-randaug-2-5-{str(num_folds)}-{len(datamodule.train_idx)}"
        logger = CSVLogger(
            log_dir,
            name=name,
        )

        # Define the callbacks
        early_stopping = EarlyStopping(monitor="val_loss", patience=12)
        model_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{name}" + "-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # Define the trainer
        trainer = Trainer(
            logger=logger,
            callbacks=[early_stopping, model_checkpoint, lr_monitor],
            max_time=timedelta(hours=4),
            log_every_n_steps=25,
            precision="16-mixed",
        )

        # Fit the model
        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

        # Test the model
        trainer.test(model, dataloaders=test_dataloader)

        # Add logger directory to list
        logger_directories.append(logger.log_dir)

    # Print the best model metrics
    printout = print_experiment_metrics(logger_directories)
    with open(f"{log_dir}/best_model_metrics.txt", "w") as file:
        file.write(printout)
