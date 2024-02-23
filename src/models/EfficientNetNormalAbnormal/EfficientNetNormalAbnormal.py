
# torch
import torch

# huggingface model
from transformers import EfficientNetForImageClassification, EfficientNetConfig

# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datetime import timedelta
import os
import sys
import torchmetrics

# Get the directory of the current script and import H5DataModule from the utilities module
# (super dirty but I can't figure out how else to make it work on Orion)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.append(os.path.abspath(project_root))
from utilities.dataset import H5DataModule  # noqa: E402


# Data file path
data_file = '/mnt/project/ngoc/CoxaAI/datasets/hips_800_sort_4.h5'

# Log file path
log_file = "/mnt/users/leobakh/VET_project/VET-Special-syllabus/logs/loss_logs"

# Model save path
model_save_path = "/mnt/users/leobakh/VET_project/VET-Special-syllabus/trained_models"


# --------------------- Model ---------------------

class EfficientNetNormalAbnormal(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the ConvNextV2 model
        config = EfficientNetConfig(num_labels=2,
                                    num_channels=1,  # For grayscale images
                                    image_size=800)
        self.model = EfficientNetForImageClassification(config)

        # Metrics
        self.accuracy = torchmetrics.Accuracy(threshold=0.5)
        self.mcc = torchmetrics.MatthewsCorrcoef(task="binary")

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self(pixel_values)
        loss = torch.nn.functional.cross_entropy(logits, labels.long())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self(pixel_values)

        # Calculate metrics
        loss = torch.nn.functional.cross_entropy(logits, labels.long())
        acc = self.accuracy(logits, labels)
        mcc = self.mcc(logits, labels)

        # Log metrics
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_mcc', mcc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


# --------------------- DataModule ---------------------

dm = H5DataModule(data_file,
                  batch_size=16,
                  train_folds=[0, 1, 2],
                  val_folds=[3],
                  test_folds=[4],
                  target_var='target',
                  tf_to_torch_channelswap=True,
                  stack_channels=False)


# --------------------- Callbacks ---------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(dirpath=model_save_path,
                                   filename='EfficientNetNormalAbnormal_best_checkpoint',
                                   monitor='val_loss',
                                   mode='min')

logger = CSVLogger(save_path=log_file,
                   name="loss_log",
                   flush_logs_every_n_steps=10)


# --------------------- Trainer ---------------------

trainer = pl.Trainer(max_time=timedelta(hours=6),
                     accelerator="gpu",
                     callbacks=[early_stopping, model_checkpoint],
                     logger=logger,
                     log_every_n_steps=25)


# --------------------- Training ---------------------

model = EfficientNetNormalAbnormal()

trainer.fit(model, dm)
