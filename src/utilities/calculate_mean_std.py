import torch
from dataset import H5DataModule


def calculate_mean_std(data_file, dm_params=None):
    if dm_params is None:
        dm_params = {}

    dm = H5DataModule(data_file,
                      batch_size=dm_params.get('batch_size', 16),
                      train_folds=dm_params.get('train_folds', [0, 1, 2]),
                      val_folds=dm_params.get('val_folds', [3]),
                      test_folds=dm_params.get('test_folds', [4]),
                      target_var=dm_params.get('target_var', 'target'),
                      tf_to_torch_channelswap=dm_params.get('tf_to_torch_channelswap', True),
                      stack_channels=dm_params.get('stack_channels', False))

    dm.setup(stage='fit')  # Setting up dataset and getting dataloader
    train_loader = dm.train_dataloader()

    # Finding channels (see if grayscale or RGB)
    first_batch, _ = next(iter(train_loader))
    channels = first_batch.size(1)

    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    nb_samples = 0

    for data, _ in train_loader:
        # Rearrange to shape [B, C, W * H] since we calculate mean and std over all pixels
        data = data.view(data.size(0), data.size(1), -1)

        # Update mean and std
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += data.size(0)

    # Finalize mean and std
    mean /= nb_samples
    std /= nb_samples

    print(f'Mean: {mean}')
    print(f'Std: {std}')
    return mean.numpy(), std.numpy()


if __name__ == '__main__':
    data_file = '/mnt/project/ngoc/CoxaAI/datasets/hips_800_sort_4.h5'
    calculate_mean_std(data_file)
