import torch # noqa
from torchvision import transforms
from PIL import Image
import numpy as np

# huggingface model
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datetime import timedelta
import os
import sys
import dotenv
dotenv.load_dotenv()

project_root = os.getenv('PROJECT_ROOT')
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal # noqa
from src.utilities.H5DataModule import H5DataModule # noqa
from src.utilities.AutoAugment.autoaugment import ImageNetPolicy # noqa


# because pytorch is dumb:
if __name__ == "__main__":
    # Model ID
    model_id = "google/vit-base-patch16-384"
    config = ViTConfig.from_pretrained(model_id)

    # Set config hyperparameters
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2

    # --------------------- Model ---------------------

    class ViTTransferNormalAbnormal(BaseNormalAbnormal):
        def __init__(self, *args, **kwargs):
            # Initialize the ConvNextV2 model with specific configuration
            model = ViTForImageClassification.from_pretrained(model_id, config=config)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
            super().__init__(model=model, *args, **kwargs)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=5e-6)

    # --------------------- Preprocessing ---------------------

    feature_extractor = ViTImageProcessor.from_pretrained(model_id)

    size = (feature_extractor.size["height"], feature_extractor.size["width"])

    def train_preprocess(image):
        # image is a numpy array in the shape (H, W, C)
        image = (image * 255).astype(np.uint8)

        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)

        # Now convert to a PIL Image
        try:
            image = Image.fromarray(image)
        except TypeError as e:
            print(f"Error converting array to image: {e}")
            # Additional debugging info
            print(f"Array shape: {image.shape}, Array dtype: {image.dtype}")
            raise

        transform_pipeline = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor()
        ])
        image = transform_pipeline(image)

        data = feature_extractor(images=image,
                                 return_tensors="pt",
                                 input_data_format="channels_first",
                                 do_rescale=True)
        data = {"pixel_values": image}
        pixel_values = data["pixel_values"]
        if pixel_values.shape[0] == 1:  # Check if the batch dimension is 1
            pixel_values = pixel_values.squeeze(0)  # Remove the first dimension
        return pixel_values

    def val_test_preprocess(image):
        # image is a numpy array in the shape (H, W, C)
        image = (image * 255).astype(np.uint8)

        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)

        # Now convert to a PIL Image
        try:
            image = Image.fromarray(image)
        except TypeError as e:
            print(f"Error converting array to image: {e}")
            # Additional debugging info
            print(f"Array shape: {image.shape}, Array dtype: {image.dtype}")
            raise

        transform_pipeline = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        image = transform_pipeline(image)

        data = feature_extractor(images=image,
                                 return_tensors="pt",
                                 input_data_format="channels_first",
                                 do_rescale=True)
        data = {"pixel_values": image}
        pixel_values = data["pixel_values"]
        if pixel_values.shape[0] == 1:  # Check if the batch dimension is 1
            pixel_values = pixel_values.squeeze(0)  # Remove the first dimension
        return pixel_values

    # --------------------- DataModule ---------------------

    dm = H5DataModule(os.getenv("DATA_FILE"),
                      batch_size=16,
                      train_folds=[0, 1, 2],
                      val_folds=[3],
                      test_folds=[4],
                      target_var='target',
                      train_transform=train_preprocess,
                      val_transform=val_test_preprocess,
                      test_transform=val_test_preprocess
                      )

    # ------------------ Instanciate model ------------------

    model = ViTTransferNormalAbnormal()
    model_class_name = model.__class__.__name__

    # --------------------- Callbacks ---------------------

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(dirpath=os.getenv("MODEL_SAVE_DIR"),
                                       filename=f'{model_class_name}_best_checkpoint' + '_{epoch:02d}_{val_loss:.2f}', # noqa
                                       monitor='val_loss',
                                       mode='min')
    log_dir = os.path.join(os.getenv("LOG_FILE_DIR"), "loss_logs")

    logger = CSVLogger(save_dir=log_dir, name=model_class_name, flush_logs_every_n_steps=10)

    # --------------------- Trainer ---------------------
    trainer = pl.Trainer(max_time=timedelta(hours=12),
                         accelerator="auto",
                         callbacks=[early_stopping, model_checkpoint],
                         logger=logger,
                         log_every_n_steps=25,
                         precision="bf16")

    # --------------------- Training ---------------------

    trainer.fit(model, dm)
