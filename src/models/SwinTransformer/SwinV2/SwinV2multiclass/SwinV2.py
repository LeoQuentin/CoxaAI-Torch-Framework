import torch # noqa
from torchvision import transforms
import os
from datetime import timedelta
import sys
import dotenv

# huggingface model
from transformers import Swinv2Config, Swinv2ForImageClassification, AutoImageProcessor
# Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, GradientAccumulationScheduler, StochasticWeightAveraging  # noqa
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
if project_root:
    sys.path.append(project_root)
from src.models.BaseMultiClass import BaseMulticlass # noqa
from src.utilities.H5DataModule import H5DataModule # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL # noqa
from src.augmentation.autoaugment import ImageNetPolicy # noqa


# because pytorch is dumb we have to do __init__:
if __name__ == "__main__":
    # Model ID
    model_id = "microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft"
    config = Swinv2Config.from_pretrained(model_id)

    # Set config hyperparameters
    config.hidden_dropout_prob = 0.3
    config.attention_probs_dropout_prob = 0.3

    # Other parameters
    size = (640, 640)  # 40x40 patches

    # Training parameters
    training_params = {
        "batch_size": 8,
        "early_stopping_patience": 40,
        "max_time_hours": 12,
        "train_folds": [0, 1, 2, 3],
        "val_folds": [4],
        "test_folds": [4],
        "log_every_n_steps": 25,
        "precision": "16-mixed"
    }

    # --------------------- Model ---------------------

    class SwinV2MultiClass640(BaseMulticlass):
        def __init__(self, *args, **kwargs):
            # Initialize the ConvNextV2 model with specific configuration
            model = Swinv2ForImageClassification.from_pretrained(model_id, config=config)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
            super().__init__(model=model, num_classes=5, *args, **kwargs)

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=5e-6, weight_decay=1e-2)
            lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=7,
                                                           verbose=True),
                            'monitor': 'val_loss',  # Specify the metric you want to monitor
                            'interval': 'epoch',
                            'frequency': 1}
            return [optimizer], [lr_scheduler]

    # --------------------- Preprocessing ---------------------

    feature_extractor = AutoImageProcessor.from_pretrained(model_id)
    feature_extractor.size = size

    def train_preprocess(image):
        # image is a numpy array in the shape (H, W, C)
        image = np_image_to_PIL(image)  # convert to PIL image

        # Preprocess the image
        transform_pipeline = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor()
        ])
        image = transform_pipeline(image)

        # Extract features using the feature extractor from Huggingface
        data = feature_extractor(images=image,
                                 return_tensors="pt",
                                 input_data_format="channels_first",
                                 do_rescale=False)  # false since transforms.ToTensor does it
        # Sometimes the feature extractor adds a batch dim
        image = data["pixel_values"]
        if len(image.size()) == 4:
            image = image.squeeze(0)
        return image

    def val_test_preprocess(image):
        # basically same as train_preprocess but without the augmentations
        image = np_image_to_PIL(image)  # convert to PIL image

        transform_pipeline = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        image = transform_pipeline(image)

        data = feature_extractor(images=image,
                                 return_tensors="pt",
                                 input_data_format="channels_first",
                                 do_rescale=False)
        image = data["pixel_values"]
        if len(image.size()) == 4:
            image = image.squeeze(0)
        return image

    # --------------------- DataModule ---------------------

    dm = H5DataModule(os.getenv("DATA_FILE"),
                      batch_size=training_params["batch_size"],
                      train_folds=training_params["train_folds"],
                      val_folds=training_params["val_folds"],
                      test_folds=training_params["test_folds"],
                      target_var='diagnosis',
                      train_transform=train_preprocess,
                      val_transform=val_test_preprocess,
                      test_transform=val_test_preprocess
                      )

    # ------------------ Instanciate model ------------------

    model = SwinV2MultiClass640()

    # log training parameters
    model.save_hyperparameters(training_params)
    model.save_hyperparameters({"size": size})

    # --------------------- Train ---------------------

    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=training_params["early_stopping_patience"])
    model_checkpoint = ModelCheckpoint(dirpath=os.getenv("MODEL_SAVE_DIR"),
                                       filename=f'{model.__class__.__name__}_best_checkpoint' + '_{epoch:02d}_{val_loss:.2f}',  # noqa
                                       monitor='val_loss',
                                       mode='min',
                                       save_top_k=1)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger
    log_dir = os.path.join(os.getenv("LOG_FILE_DIR"), "loss_logs")
    logger = CSVLogger(save_dir=log_dir,
                       name=model.__class__.__name__,
                       flush_logs_every_n_steps=training_params["log_every_n_steps"])

    # Trainer
    trainer = Trainer(max_time=timedelta(hours=training_params["max_time_hours"]),
                      accelerator="auto",
                      callbacks=[early_stopping, model_checkpoint, lr_monitor],  # noqa
                      logger=logger,
                      log_every_n_steps=training_params["log_every_n_steps"],
                      precision=training_params["precision"])

    # Training
    trainer.fit(model, dm)

    # Best model path
    best_model_path = model_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")
