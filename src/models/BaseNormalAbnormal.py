# torch
import torch

# Lightning
import pytorch_lightning as pl
import torchmetrics


# --------------------- Model ---------------------

class BaseNormalAbnormal(pl.LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        # Initialize the ConvNextV2 model
        self.model = model

        # Metrics
        # self.accuracy = torchmetrics.Accuracy(threshold=0.5, task="binary")
        # self.mcc = torchmetrics.MatthewsCorrCoef(task="binary")

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
        # acc = self.accuracy(logits, labels)
        # mcc = self.mcc(logits, labels)

        # Log metrics
        self.log('val_loss', loss)
        # self.log('val_acc', acc)
        # self.log('val_mcc', mcc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)
