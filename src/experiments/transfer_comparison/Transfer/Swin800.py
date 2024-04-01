import torch  # noqa
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from datetime import timedelta

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# huggingface model
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor

import os
import sys
import dotenv

dotenv.load_dotenv()

project_root = os.getenv("PROJECT_ROOT")
if project_root:
    sys.path.append(project_root)
from src.experiments.transfer_comparison.preprocess_funcs import (  # noqa
    train_augments,
    val_test_augments,
)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal  # noqa
from src.models.SimpleTrainingLoop import train_model  # noqa
from src.utilities.H5DataModule import H5DataModule  # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL  # noqa
from src.augmentation.autoaugment import ImageNetPolicy  # noqa

log_dir = os.path.join(
    project_root, "src/experiments/transfer_comparison/Transfer/logs"
)

checkpoint_dir = os.path.join(
    project_root, "src/experiments/transfer_comparison/Transfer/checkpoints"
)

experiment_file_name = "ViT800_Transfer"

# because pytorch is dumb we have to do __init__:
if __name__ == "__main__":
    # Model ID
    model_id = "google/vit-base-patch16-384"
    config = AutoConfig.from_pretrained(model_id)

    # Set config hyperparameters
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2

    # Other parameters
    size = (800, 800)  # 40x40 patches

    # Training parameters
    training_params = {
        "batch_size": 8,
        "early_stopping_patience": 10,
        "max_time_hours": 20,
        "train_folds": [0, 1, 2],
        "val_folds": [3],
        "test_folds": [4],
        "lr_scheduler_factor": 0.2,
        "lr_scheduler_patience": 5,
        "log_every_n_steps": 10,
        "precision": "16-mixed",
    }

    # --------------------- Model ---------------------

    class NeuralNetwork(BaseNormalAbnormal):
        def __init__(self, *args, **kwargs):
            # Initialize the ConvNextV2 model with specific configuration
            model = AutoModelForImageClassification.from_pretrained(model_id, config=config)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
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

        def forward(self, pixel_values):
            outputs = self.model(
                pixel_values=pixel_values, interpolate_pos_encoding=True
            )
            return outputs.logits

    # --------------------- Preprocessing ---------------------

    feature_extractor = AutoImageProcessor.from_pretrained(model_id)
    feature_extractor.size = size

    def train_preprocess(image):
        image = train_augments(image, size=(800, 800))

        # Extract features using the feature extractor from Huggingface
        data = feature_extractor(
            images=image,
            return_tensors="pt",
            input_data_format="channels_first",
            do_rescale=False,
        )  # false since transforms.ToTensor does it
        # Sometimes the feature extractor adds a batch dim
        image = data["pixel_values"]
        if len(image.size()) == 4:
            image = image.squeeze(0)
        return image

    def val_test_preprocess(image):
        # basically same as train_preprocess but without the augmentations
        image = val_test_augments(image, size=(800, 800))

        data = feature_extractor(
            images=image,
            return_tensors="pt",
            input_data_format="channels_first",
            do_rescale=False,
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

    model = NeuralNetwork()

    # ------------------ Trainer ------------------

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=training_params["early_stopping_patience"]
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=os.getenv("MODEL_SAVE_DIR"),
        filename=f"{experiment_file_name}_best_checkpoint"
        + "_{epoch:02d}_{val_loss:.2f}",  # noqa
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = CSVLogger(
        save_dir=log_dir,
        name=experiment_file_name,
        flush_logs_every_n_steps=training_params["log_every_n_steps"],
    )

    # Trainer
    trainer = Trainer(
        max_time=timedelta(hours=training_params["max_time_hours"]),
        accelerator="auto",
        callbacks=[
            early_stopping,
            model_checkpoint,
            lr_monitor,
        ],
        logger=logger,
        log_every_n_steps=training_params["log_every_n_steps"],
        precision=training_params["precision"],
    )

    # Training
    trainer.fit(model, dm)

    # Best model path
    best_model_path = model_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")

    # ----------------------- Test -----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the test dataloader
    dm.setup(stage="test")
    test_dataloader = dm.test_dataloader()

    # Initialize variables to store predictions and labels
    all_predictions = []
    all_labels = []

    # Disable gradient computation
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch

            # Move inputs and labels to the appropriate device (e.g., GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Run inference on the test batch
            outputs = model(inputs)

            # Get the predicted labels
            _, predicted = torch.max(outputs, 1)

            # Append predictions and labels to the lists
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    with open("transfer_metrics.txt", "a") as file:
        file.write(f"Model: {model_id}, Image Size: {size}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n")
        file.write("\n")
        file.flush()
