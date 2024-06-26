# Description: PyTorch Dataset and DataModule classes for loading H5 datasets to PyTorch models.
from torch.utils.data import DataLoader, Dataset
import h5py
import torch
import pytorch_lightning as pl
import dotenv
import os

dotenv.load_dotenv()


class H5FoldDataset(Dataset):
    def __init__(self, file_path, folds, target_var="target", transform=None):
        """
        Initialize the Dataset object for loading data from specified folds in an H5 file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the dataset.
        folds : list of int
            A list of integers representing the folds of the dataset to fetch.
        target_var : str, optional
            The name of the target variable, by default 'target'.
        transform : callable, optional
            A function/transform to apply to the data, by default None.
        tf_to_torch_channelswap : bool, optional
            Whether to swap the channels of the images to:
            PyTorch format: (C, H, W), from TensorFlow format: (H, W, C),
            by default True.
        stack_channels : bool, optional
            Whether to stack the channels of the images to create a 3-channel image,
            by default False.
        """
        self.file_path = file_path
        self.folds = folds
        self.target_var = target_var
        self.transform = transform
        with h5py.File(self.file_path, "r") as h5_file:
            self.lengths = [
                h5_file[f"fold_{fold}/image"].shape[0] for fold in self.folds
            ]
            self.index_mapping = []
            for i, fold in enumerate(self.folds):
                for j in range(self.lengths[i]):
                    self.index_mapping.append((fold, j))

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        fold, index = self.index_mapping[idx]
        with h5py.File(self.file_path, "r") as h5_file:
            image = h5_file[f"fold_{fold}/image"][index]
            target = h5_file[f"fold_{fold}/{self.target_var}"][
                index
            ]  # Use the specified target

            # Apply transform if specified
            if self.transform:
                image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32)


class H5DataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading and preparing H5 datasets for model training and evaluation.

    Parameters
    ----------
    data_file : str
        File path of the H5 dataset.
    train_folds : list of int
        List of integers specifying the training folds.
    val_fold : list of int
        List of integers specifying the validation fold.
    test_fold : list of int
        List of integers specifying the testing fold.
    batch_size : int
        Batch size for the dataloaders.
    transform : callable, optional
        Optional transform to be applied on a sample, by default None.
    tf_to_torch_channelswap : bool, optional
            Whether to swap the channels of the images to:
            PyTorch format: (C, H, W), from TensorFlow format: (H, W, C),
            by default True.
    stack_channels : bool, optional
        Whether to stack the single-channel data to create 3-channel images, by default False.
    target_var : str, optional
        Name of the target variable in the dataset ('target' or 'diagnosis'), by default 'target'.
    """

    def __init__(
        self,
        data_file,
        batch_size=16,
        train_folds=None,
        val_folds=None,
        test_folds=None,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        target_var="target",
        train_loader_workers=None,
        val_loader_workers=None,
        test_loader_workers=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_file = data_file
        self.batch_size = batch_size
        self.train_folds = train_folds if train_folds is not None else [0, 1, 2]
        self.val_folds = val_folds if val_folds is not None else [4]
        self.test_folds = test_folds if test_folds is not None else [3]
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
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

    def setup(self, stage=None):
        """
        Initializes datasets for different stages of training, validation, and testing.

        Parameters
        ----------
        stage : str, optional
            Specifies the stage for which to set up the data.
            Can be 'fit', 'validate', 'test', or None.
            If None, datasets for all stages are initialized.
        """
        common_params = {
            "file_path": self.data_file,
            "target_var": self.target_var,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = H5FoldDataset(
                **common_params, folds=self.train_folds, transform=self.train_transform
            )
            self.val_dataset = H5FoldDataset(
                **common_params, folds=self.val_folds, transform=self.val_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = H5FoldDataset(
                **common_params, folds=self.test_folds, transform=self.test_transform
            )

    def train_dataloader(self):
        """
        Creates a DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            A DataLoader instance configured for the training dataset, with shuffling enabled.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_loader_workers,
        )

    def val_dataloader(self):
        """
        Creates a DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            A DataLoader instance configured for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.val_loader_workers,
        )

    def test_dataloader(self):
        """
        Creates a DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader instance configured for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.test_loader_workers,
        )

    @staticmethod
    def from_base_config(override=None):
        """
        Creates an instance of H5DataModule using the default configuration.
        Can override any values in given configuration by passing a dictionary
        with the desired parameters.

        default configuration:
        - data_file: DATA_FILE environment variable
        - batch_size: 16
        - train_folds: [0, 1, 2]
        - val_folds: [3]
        - test_folds: [4]
        - target_var: 'target'
        - train_transform: None
        - val_transform: None
        - test_transform: None
        - train_loader_workers: 16
        - val_loader_workers: 8
        - test_loader_workers: 6

        Returns
        -------
        H5DataModule
            An instance of H5DataModule with default configuration.
        """
        if not override:
            override = {}

        return H5DataModule(
            data_file=override.get("data_file", os.getenv("DATA_FILE")),
            batch_size=override.get("batch_size", 16),
            train_folds=override.get("train_folds", [0, 1, 2]),
            val_folds=override.get("val_folds", [3]),
            test_folds=override.get("test_folds", [4]),
            target_var=override.get("target_var", "target"),
            train_transform=override.get("train_transform", None),
            val_transform=override.get("val_transform", None),
            test_transform=override.get("test_transform", None),
            train_loader_workers=override.get("train_loader_workers", 16),
            val_loader_workers=override.get("val_loader_workers", 8),
            test_loader_workers=override.get("test_loader_workers", 6),
        )
