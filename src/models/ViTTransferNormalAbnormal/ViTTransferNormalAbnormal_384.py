import torch # noqa
from torchvision import transforms

# huggingface model
from transformers import ViTImageProcessor, ViTForImageClassification
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


# Model ID
model_id = "google/vit-base-patch16-384"


# --------------------- Model ---------------------

class ViTTransferNormalAbnormal(BaseNormalAbnormal):
    def __init__(self, *args, **kwargs):
        # Initialize the ConvNextV2 model with specific configuration
        model = ViTForImageClassification.from_pretrained(model_id)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
        super().__init__(model=model, *args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)


# --------------------- Preprocessing ---------------------

feature_extractor = ViTImageProcessor.from_pretrained(model_id)


def preprocess_image(image: torch.Tensor):
    data = feature_extractor(images=image,
                             return_tensors="pt",
                             input_data_format="channels_first",
                             do_rescale=False)
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
                  tf_to_torch_channelswap=True,
                  stack_channels=True,
                  train_transform=preprocess_image,
                  val_transform=preprocess_image,
                  test_transform=preprocess_image
                  )


# ------------------ Instanciate model ------------------

model = ViTTransferNormalAbnormal()
model_class_name = model.__class__.__name__


# --------------------- Callbacks ---------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=20)
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
                     log_every_n_steps=25)


# --------------------- Training ---------------------

trainer.fit(model, dm)
