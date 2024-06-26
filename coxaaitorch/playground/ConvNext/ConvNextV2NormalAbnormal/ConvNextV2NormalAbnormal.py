import torch  # noqa
import torch.nn as nn
from torchvision import transforms
import os
from datetime import timedelta
import sys
import dotenv

# huggingface model
from transformers import (
    AutoImageProcessor,
    ConvNextV2ForImageClassification,
    ConvNextV2Config,
)

# Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

dotenv.load_dotenv()
project_root = os.getenv("PROJECT_ROOT")
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal  # noqa
from src.models.SimpleTrainingLoop import train_model  # noqa
from src.utilities.H5DataModule import H5DataModule  # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL  # noqa
from src.augmentation.autoaugment import ImageNetPolicy  # noqa


# because pytorch is dumb we have to do __init__:
if __name__ == "__main__":
    # Model ID
    model_id = "facebook/convnextv2-tiny-22k-384"
    config = ConvNextV2Config.from_pretrained(model_id)

    # Set config hyperparameters
    # ---------------------

    # Other parameters
    size = (384, 384)

    # Training parameters
    training_params = {
        "batch_size": 16,
        "early_stopping_patience": 20,
        "max_time_hours": 12,
        "train_folds": [0, 1, 2, 3],
        "val_folds": [4],
        "test_folds": [4],
        "log_every_n_steps": 25,
        "precision": 16,
    }

    # --------------------- Model ---------------------

    class ConvNextV2NormalAbnormal(BaseNormalAbnormal):
        def __init__(self, *args, **kwargs):
            # Initialize the ConvNextV2 model with specific configuration
            model = ConvNextV2ForImageClassification(config)
            model.classifier = nn.Sequential(
                nn.Linear(
                    config.hidden_sizes[-1], 512
                ),  # First layer to 512 hidden nodes
                nn.ReLU(),  # ReLU activation function
                nn.Linear(512, 2),  # Second layer to the final output
            )

            super().__init__(model=model, *args, **kwargs)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=3e-4)

    # --------------------- Preprocessing ---------------------

    feature_extractor = AutoImageProcessor.from_pretrained(model_id)

    def train_preprocess(image):
        # image is a numpy array in the shape (H, W, C)
        image = np_image_to_PIL(image)  # convert to PIL image

        # Preprocess the image
        transform_pipeline = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
            ]
        )
        image = transform_pipeline(image)

        # Extract features using the feature extractor from Huggingface
        data = feature_extractor(
            images=image,
            return_tensors="pt",
            input_data_format="channels_first",
            do_rescale=False,  # false since transforms.ToTensor does it
            do_resize=False,
        )
        # Sometimes the feature extractor adds a batch dim
        image = data["pixel_values"]
        if len(image.size()) == 4:
            image = image.squeeze(0)
        return image

    def val_test_preprocess(image):
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

        data = feature_extractor(
            images=image,
            return_tensors="pt",
            input_data_format="channels_first",
            do_rescale=False,
            do_resize=False,
        )
        image = data["pixel_values"]
        if len(image.size()) == 4:
            image = image.squeeze(0)
        return image

    # --------------------- DataModule ---------------------

    dm = H5DataModule(
        os.getenv("DATA_FILE"),
        batch_size=training_params["batch_size"],
        train_folds=training_params["train_folds"],
        val_folds=training_params["val_folds"],
        test_folds=training_params["test_folds"],
        target_var="target",
        train_transform=train_preprocess,
        val_transform=val_test_preprocess,
        test_transform=val_test_preprocess,
    )

    # ------------------ Instanciate model ------------------

    model = ConvNextV2NormalAbnormal()

    # log training parameters
    model.save_hyperparameters(training_params)
    model.save_hyperparameters({"size": size})

    # --------------------- Train ---------------------

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=training_params["early_stopping_patience"]
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=os.getenv("MODEL_SAVE_DIR"),
        filename=f"{model.__class__.__name__}_best_checkpoint"
        + "_{epoch:02d}_{val_loss:.2f}",  # noqa
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Logger
    log_dir = os.path.join(os.getenv("LOG_FILE_DIR"), "loss_logs")
    logger = CSVLogger(
        save_dir=log_dir,
        name=model.__class__.__name__,
        flush_logs_every_n_steps=training_params["log_every_n_steps"],
    )

    # Trainer
    trainer = Trainer(
        max_time=timedelta(hours=training_params["max_time_hours"]),
        accelerator="auto",
        callbacks=[early_stopping, model_checkpoint],
        logger=logger,
        log_every_n_steps=training_params["log_every_n_steps"],
        precision=training_params["precision"],
    )

    # Training
    trainer.fit(model, dm)

    # Best model path
    best_model_path = model_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")
