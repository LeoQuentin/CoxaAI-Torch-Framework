import torch
import argparse
import os
import dotenv
from coxaaitorch.utilities import H5DataModule
import json

dotenv.load_dotenv()


def calculate_mean_std(train_loader):
    # Assuming train_loader is a PyTorch DataLoader
    channels = train_loader.dataset[0][0].size(0)  # Tensor of first image

    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    nb_samples = 0

    for data, _ in train_loader:
        data = data.view(
            data.size(0), data.size(1), -1
        )  # Rearrange data for mean/std calculation
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += data.size(0)

    mean /= nb_samples
    std /= nb_samples

    print(f"Mean: {mean}")
    print(f"Std: {std}")
    return mean.numpy(), std.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mean and standard deviation of dataset."
    )  # noqa
    parser.add_argument(
        "--data_file",
        type=str,
        default=os.getenv("DATA_FILE"),
        help="Path to the data file. Defaults to DATA_FILE environment variable.",
    )
    parser.add_argument(
        "--dm_params", type=str, help="DataModule parameters as a JSON string."
    )
    args = parser.parse_args()

    if not args.data_file:
        raise ValueError(
            "No data file provided and DATA_FILE environment variable is not set."
        )

    # Initialize H5DataModule with command-line arguments, environment variables, or default values
    dm_params = {
        "batch_size": 16,
        "train_folds": [0, 1, 2],
        "val_folds": [3],
        "test_folds": [4],
        "target_var": "target",
        "tf_to_torch_channelswap": True,
        "stack_channels": False,
    }

    if args.dm_params:
        try:
            command_line_params = json.loads(args.dm_params)
            dm_params.update(
                command_line_params
            )  # Update default params with any provided via cli
        except json.JSONDecodeError:
            raise ValueError("Failed to parse --dm_params as JSON.")

    dm = H5DataModule(args.data_file, **dm_params)

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    calculate_mean_std(train_loader)
