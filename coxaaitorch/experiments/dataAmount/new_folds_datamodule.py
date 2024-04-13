import torch
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from coxaaitorch.utilities import H5FoldDataset
import numpy as np
import os
import dotenv

dotenv.load_dotenv()


initial_dataset = H5FoldDataset(
    file_path=os.getenv("DATA_FILE"), folds=[1, 2, 3], target_var="diagnosis"
)

test_dataset = H5FoldDataset(
    file_path=os.getenv("DATA_FILE"), folds=[4], target_var="diagnosis"
)

val_dataset = H5FoldDataset(
    file_path=os.getenv("DATA_FILE"), folds=[3], target_var="diagnosis"
)


class NewFoldsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        val_dataset,
        test_dataset,
        batch_size=32,
        dataset_splits=20,
        used_folds=1,
        target_var="target",
        train_loader_workers=None,
        val_loader_workers=None,
        test_loader_workers=None,
        random_state=42,
    ):
        super().__init__()
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.dataset_splits = dataset_splits
        self.used_folds = used_folds
        self.random_state = random_state
        self.target_var = target_var
        self.train_loader_workers = (
            train_loader_workers
            if train_loader_workers
            else os.getenv("TRAIN_LOADER_WORKERS", 4)
        )
        self.val_loader_workers = (
            val_loader_workers
            if val_loader_workers
            else os.getenv("VAL_LOADER_WORKERS", 4)
        )
        self.test_loader_workers = (
            test_loader_workers
            if test_loader_workers
            else os.getenv("TEST_LOADER_WORKERS", 2)
        )

    def prepare_data(self):
        self.indices = np.arange(len(self.dataset))
        self.targets = [target for _, target in self.dataset]
        self.skf = StratifiedKFold(
            n_splits=self.dataset_splits, shuffle=True, random_state=self.random_state
        )
        folds = list(self.skf.split(self.indices, self.targets))
        self.folds = [fold[1] for fold in folds]

    def setup(self, stage=None):
        train_idx = self.folds[: self.used_folds]
        self.train_idx = np.concatenate(train_idx)
        self.train_dataset.target_var = self.target_var
        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_idx)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.train_loader_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.val_loader_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.test_loader_workers,
            shuffle=False,
        )
