# Lightning Module that is used for binary classification tasks. Subclass to create a binary model.
import torch
import torch.nn as nn

# Lightning
import pytorch_lightning as pl
import torchmetrics


# --------------------- Model ---------------------


class BaseNetwork(pl.LightningModule):
    def __init__(self, model, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model"])

        # Store the model as an attribute
        self.model = model
        self.loss = None  # set with method configure_loss_func
        self.configure_loss_func()

        # Multi-class or binary
        self.num_classes = num_classes
        self.metric_task = "binary" if num_classes == 2 else "multiclass"

        # Metrics
        self.metrics = {}  # set with method set_metrics
        self.set_metrics()

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
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
        preds = torch.argmax(logits, dim=1)

        # Move metrics to device
        self.metrics_to_device()

        # Calculate metrics
        loss = self.loss(logits, labels.long())
        for name, metric in self.metrics.items():
            metric_value = metric(preds, labels)
            self.log(f"val_{name}", metric_value)

        # Log metrics
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self(pixel_values)
        preds = torch.argmax(logits, dim=1)

        # Move metrics to device
        self.metrics_to_device()

        loss = self.loss(logits, labels.long())

        # Calculate and log loss and metrics
        self.log("test_loss", loss)

        test_metrics = {}
        # All other metrics
        for name, metric in self.metrics.items():
            metric_value = metric(preds, labels)
            test_metrics[f"test_{name}"] = metric_value
            self.log(f"test_{name}", metric_value)

        return test_metrics

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

    def set_metrics(self):
        common_settings = {
            "task": self.metric_task,
            "num_classes": self.num_classes,
        }
        self.metrics = {
            "accuracy": torchmetrics.Accuracy(**common_settings),
            "precision": torchmetrics.Precision(**common_settings, average="macro"),
            "recall": torchmetrics.Recall(**common_settings, average="macro"),
            "specificity": torchmetrics.Specificity(**common_settings),
            "f1": torchmetrics.F1Score(**common_settings, average="macro"),
            "mcc": torchmetrics.MatthewsCorrCoef(**common_settings),
        }
        self.metrics_to_device()
