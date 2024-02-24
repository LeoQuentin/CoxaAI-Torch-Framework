import torch # noqa

# huggingface model
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config

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


# --------------------- Model ---------------------


class ConvNextV2NormalAbnormal(BaseNormalAbnormal):
    def __init__(self, *args, **kwargs):
        # Initialize the ConvNextV2 model with specific configuration
        convnext_v2_config = ConvNextV2Config(num_labels=2,
                                              num_channels=1,  # For grayscale images
                                              image_size=800)  # Customize your config here
        convnext_v2_model = ConvNextV2ForImageClassification(convnext_v2_config)

        super().__init__(model=convnext_v2_model, *args, **kwargs)

# --------------------- DataModule ---------------------


dm = H5DataModule(os.getenv("DATA_FILE"),
                  batch_size=16,
                  train_folds=[0, 1, 2],
                  val_folds=[3],
                  test_folds=[4],
                  target_var='target',
                  tf_to_torch_channelswap=True,
                  stack_channels=False)


# --------------------- Callbacks ---------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(dirpath=os.getenv("DATA_FILE"),
                                   filename='ConvNextV2NormalAbnormal_best_checkpoint',
                                   monitor='val_loss',
                                   mode='min')
log_dir = os.path.join(os.getenv("LOG_FILE_DIR"), "loss_logs")

logger = CSVLogger(save_dir=log_dir, name="ConvNextV2NormalAbnormal", flush_logs_every_n_steps=10)


# --------------------- Trainer ---------------------
trainer = pl.Trainer(max_time=timedelta(hours=6),
                     accelerator="auto",
                     callbacks=[early_stopping, model_checkpoint],
                     logger=logger,
                     log_every_n_steps=25)


# --------------------- Training ---------------------

model = ConvNextV2NormalAbnormal()

trainer.fit(model, dm)
