
# torch
import torch
from torch.utils.data import DataLoader

# huggingface model
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config

# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datetime import timedelta
import os
import sys

# Get the directory of the current script and import H5DataModule from the utilities module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.append(os.path.abspath(project_root))
from utilities.dataset import H5DataModule


# Data file path
data_file = '/mnt/project/ngoc/CoxaAI/datasets/hips_800_sort_4.h5'


# --------------------- Model ---------------------

class NormalAbnormalConvNextV2(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the ConvNextV2 model
        config = ConvNextV2Config(num_labels=2,
                                  num_channels=1,  # For grayscale images
                                  image_size=800,
                                  out_features=["logits"])
        self.model = ConvNextV2ForImageClassification(config)

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
        loss = torch.nn.functional.cross_entropy(logits, labels.long())
        self.log('val_loss', loss)
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
model_checkpoint = ModelCheckpoint(dirpath='./models/',
                                   filename='best-model',
                                   monitor='val_loss',
                                   mode='min')
logger = CSVLogger("loss_log", name="my_model", flush_logs_every_n_steps=10)


# --------------------- Trainer ---------------------
trainer = pl.Trainer(max_time=timedelta(hours=6),
                     callbacks=[early_stopping, model_checkpoint],
                     logger=logger,
                     gpus=1 if torch.cuda.is_available() else 0,  # Adjust based on your setup
                     log_every_n_steps=10)


# --------------------- Training ---------------------

model = NormalAbnormalConvNextV2()

trainer.fit(model, dm)
