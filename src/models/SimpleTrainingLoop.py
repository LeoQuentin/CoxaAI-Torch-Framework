import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def train_model(dm,
                model,
                max_time_hours=12,
                early_stopping_patience=10,
                log_every_n_steps=25,
                presicion="32-true"):
    """
    Trains the given model using the provided DataModule.

    Parameters:
    - dm: The DataModule containing the training, validation, and test datasets.
    - model: The model to be trained.
    - max_time_hours: Maximum time in hours for training.
    - batch_size: Batch size for training.
    - early_stopping_patience: Number of epochs with no improvement after which training stops.
    - log_every_n_steps: Number of steps after which to log training progress.

    Returns:
    - trainer: The PyTorch Lightning Trainer object after training.
    - best_model_path: Path to the best model checkpoint.
    """
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
    model_checkpoint = ModelCheckpoint(dirpath=os.getenv("MODEL_SAVE_DIR"),
                                       filename=f'{model.__class__.__name__}_best_checkpoint' + '_{epoch:02d}_{val_loss:.2f}',  # noqa
                                       monitor='val_loss',
                                       mode='min',
                                       save_top_k=1)

    # Logger
    log_dir = os.path.join(os.getenv("LOG_FILE_DIR"), "loss_logs")
    logger = CSVLogger(save_dir=log_dir,
                       name=model.__class__.__name__,
                       flush_logs_every_n_steps=log_every_n_steps)

    # Trainer
    trainer = Trainer(max_time={'hours': max_time_hours},
                      accelerator="auto",
                      callbacks=[early_stopping, model_checkpoint],
                      logger=logger,
                      log_every_n_steps=log_every_n_steps,
                      precision=presicion)

    # Training
    trainer.fit(model, dm)

    # Best model path
    best_model_path = model_checkpoint.best_model_path

    return trainer, best_model_path
