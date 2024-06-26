import torch  # noqa
from torchvision import transforms
from PIL import Image
import numpy as np

# huggingface model
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig

# Lightning
import os
import sys
import dotenv

dotenv.load_dotenv()

project_root = os.getenv("PROJECT_ROOT")
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal  # noqa
from src.models.SimpleTrainingLoop import train_model  # noqa
from src.utilities.H5DataModule import H5DataModule  # noqa
from src.augmentation.autoaugment import ImageNetPolicy  # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL  # noqa


# because pytorch is dumb we have to do __init__:
if __name__ == "__main__":
    # Model ID
    model_id = "google/vit-base-patch16-384"
    config = ViTConfig.from_pretrained(model_id)

    # Set config hyperparameters
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1

    # Training parameters
    training_params = {
        "batch_size": 8,
        "early_stopping_patience": 10,
        "max_time_hours": 12,
        "train_folds": [0, 1, 2],
        "val_folds": [3],
        "test_folds": [4],
    }

    # --------------------- Model ---------------------

    class ViTTransferNormalAbnormal(BaseNormalAbnormal):
        def __init__(self, *args, **kwargs):
            # Initialize the ConvNextV2 model with specific configuration
            model = ViTForImageClassification.from_pretrained(model_id, config=config)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
            super().__init__(model=model, *args, **kwargs)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=5e-6)

        def forward(self, pixel_values):
            outputs = self.model(
                pixel_values=pixel_values, interpolate_pos_encoding=True
            )
            return outputs.logits

    # --------------------- Preprocessing ---------------------

    feature_extractor = ViTImageProcessor.from_pretrained(model_id)
    size = (384, 384)
    feature_extractor.size = size

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
            do_rescale=False,
        )  # false since transforms.ToTensor does it
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

    model = ViTTransferNormalAbnormal()

    # log training parameters
    model.save_hyperparameters(training_params)

    # --------------------- Train ---------------------

    accepted_params = ["early_stopping_patience", "max_time_hours"]  # noqa
    training_params = {k: v for k, v in training_params.items() if k in accepted_params}

    trainer, path_to_bet_model = train_model(dm, model, **training_params)
