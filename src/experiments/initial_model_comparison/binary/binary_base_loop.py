import torch
from torchvision import transforms
import os
from datetime import timedelta
import sys
import dotenv

from transformers import AutoConfig, AutoModelForImageClassification
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal  # noqa 
from src.utilities.H5DataModule import H5DataModule  # noqa 
from src.utilities.np_image_to_PIL import np_image_to_PIL  # noqa 
from src.augmentation.autoaugment import ImageNetPolicy  # noqa 


def run_training(training_params):
    config = AutoConfig.from_pretrained(model_id)
    config.image_size = training_params["size"]
    config.num_channels = training_params["num_channels"]

    class EfficientNet(BaseNormalAbnormal):
        def __init__(self, *args, **kwargs):
            model = AutoModelForImageClassification.from_config(config)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
            super().__init__(model=model, *args, **kwargs)

            # Training parameters
            self.learning_rate = 3e-4

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=training_params["lr_scheduler_factor"],  # noqa 
                                                           patience=training_params["lr_scheduler_patience"]),  # noqa 
                            'monitor': 'val_loss',
                            'interval': 'epoch',
                            'frequency': 1}
            return [optimizer], [lr_scheduler]

    model = EfficientNet()

    def train_preprocess(image):
        image = np_image_to_PIL(image)
        transform_pipeline = training_params["train_transform"]  # augmentations
        image = transform_pipeline(image)
        if len(image.size()) == 4:  # remove batch dim that sometimes shows up
            image = image.squeeze(0)
        return image

    def val_test_preprocess(image):
        image = np_image_to_PIL(image)
        transform_pipeline = training_params["val_test_transform"]  # augmentations
        image = transform_pipeline(image)
        if len(image.size()) == 4:  # remove batch dim that sometimes shows up
            image = image.squeeze(0)
        return image

    dm = H5DataModule(os.getenv("DATA_FILE"),
                      batch_size=training_params["batch_size"],
                      train_folds=training_params["train_folds"],
                      val_folds=training_params["val_folds"],
                      test_folds=training_params["test_folds"],
                      target_var='target',
                      train_transform=train_preprocess,
                      val_transform=val_test_preprocess,
                      test_transform=val_test_preprocess
                      )

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=training_params["early_stopping_patience"])

    checkpoint_dir = training_params["checkpoint_path"]
    model_checkpoint = ModelCheckpoint(dirpath=checkpoint_dir,
                                       filename=f'{model_id}_binary_{training_params["log_checkpoint_description"]}_{size[0]}_best_checkpoint' + '_{epoch:02d}_{val_loss:.2f}',  # noqa 
                                       monitor='val_loss',
                                       mode='min',
                                       save_top_k=1)

    log_dir = training_params["log_path"]
    logger = CSVLogger(save_dir=log_dir,
                       name=f"{model_id}_{training_params['log_checkpoint_description']}_{size[0]}",
                       flush_logs_every_n_steps=training_params["log_every_n_steps"])

    trainer = Trainer(max_time=timedelta(hours=training_params["max_time_hours"]),
                      accelerator="auto",
                      callbacks=[early_stopping,
                                 model_checkpoint,
                                 LearningRateMonitor(logging_interval='step')],
                      logger=logger,
                      log_every_n_steps=training_params["log_every_n_steps"],
                      precision=training_params["presicion"])

    trainer.fit(model, dm)

    best_model_path = model_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")


if __name__ == "__main__":
    for model_name in ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3",
                       "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]:
        model_id = f"google/{model_name}"
        size = (384, 384)
        training_params = {
            "model_id": model_id,
            "batch_size": (32 if model_name in ["efficientnet-b0", "efficientnet-b1",
                                                "efficientnet-b2", "efficientnet-b3"] else 16),
            "early_stopping_patience": 10,
            "max_time_hours": 12,
            "train_folds": [0, 1, 2],
            "val_folds": [3],
            "test_folds": [4],
            "log_every_n_steps": 10,
            "presicion": "16-mixed",
            "size": size,
            "num_channels": 1,
            "lr_scheduler_factor": 0.2,
            "lr_scheduler_patience": 5,
            "train_transform": transforms.Compose([
                transforms.Resize(size),
                transforms.RandomRotation(10),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            "val_test_transform": transforms.Compose([
                transforms.Resize(size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ]),
            "log_checkpoint_description": "lightAugReg",
            "log_path": os.path.join(project_root, "src/experiments/logs"),
            "checkpoint_path": os.path.join(project_root, "src/experiments/modelcheckpoints")
        }
        run_training(model_id, size, training_params)
