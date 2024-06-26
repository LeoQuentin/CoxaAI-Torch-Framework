# torch
import torch
import torch.nn as nn

# Lightning
import pytorch_lightning as pl
import torchmetrics


# --------------------- Model ---------------------


class BaseMulticlass(pl.LightningModule):
    def __init__(self, model, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model"])

        # Initialize the ConvNextV2 model
        self.model = model
        self.num_classes = num_classes

        self.loss = (
            nn.CrossEntropyLoss()
        )  # Placeholder to show that there is a self.loss
        self.configure_loss_func()

        # Metrics
        device = self.device  # need to manually set device
        self.metrics = {
            "accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ),
            "precision": torchmetrics.Precision(
                task="multiclass", average="macro", num_classes=num_classes
            ),  # noqa
            "recall": torchmetrics.Recall(
                task="multiclass", average="macro", num_classes=num_classes
            ),  # noqa
            "f1": torchmetrics.F1Score(
                task="multiclass", average="macro", num_classes=num_classes
            ),
            "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            "mcc": torchmetrics.MatthewsCorrCoef(
                task="multiclass", num_classes=num_classes
            ),
        }
        for name, metric in self.metrics.items():
            metric.to(device)
        # Confusion Matrix has to be dealt with differently so it's not in the dictionary.
        self.confusionMatrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )  # noqa

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self(pixel_values)
        loss = self.loss(logits, labels.long())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self(pixel_values)
        pred_labels = torch.argmax(logits, dim=1)

        # Move metrics to device
        self.metrics_to_device()

        # Calculate metrics
        loss = self.loss(logits, labels.long())
        acc = self.metrics["accuracy"](pred_labels, labels)

        # Log metrics
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        # Move metrics to device
        self.metrics_to_device()

        loss = self.loss(logits, y)

        # Calculate and log loss and conf matrix
        self.log("test_loss", loss)
        self.confusionMatrix(torch.argmax(logits, dim=1), y)

        # All other metrics
        for name, metric in self.metrics.items():
            self.log(f"test_{name}", metric(preds, y))

        return loss

    def on_test_epoch_end(self):
        # At the end of the test, log the confusion matrix as a list
        confmat_tensor = (
            self.confusionMatrix.compute()
        )  # Get the confusion matrix as a tensor
        confmat_list = confmat_tensor.numpy().tolist()  # Convert to a list
        print(
            "Confusion Matrix:", confmat_list
        )  # Print or log the confusion matrix list
        self.logger.log_metrics(
            {"confusion_matrix": confmat_list}
        )  # Log using the logger
        self.confusionMatrix.reset()  # Reset for the next use

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def on_train_start(self):
        # Save optimizer and learning rate
        self.save_hyperparameters({"optimizer": self.optimizers().__class__.__name__})
        self.save_hyperparameters({"lr": self.optimizers().param_groups[0]["lr"]})

    def configure_loss_func(self):
        """To overwrite loss function when creating modules that subclass from this."""
        self.loss = nn.CrossEntropyLoss()

    def metrics_to_device(self):
        """Move all metrics to the current device."""
        device = self.device
        for metric in self.metrics.values():
            metric.to(device)
        self.confusionMatrix.to(device)
