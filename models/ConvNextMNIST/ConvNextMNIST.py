
import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to get the root of the project
project_root = os.path.join(script_dir, '..', '..')

# Add the project root to the Python path
sys.path.append(os.path.abspath(project_root))

# Now you can import your module
from utilities.dataset import H5DataModule

# The rest of your script...

# Update this path to your actual H5 dataset file
data_file = '/mnt/project/ngoc/CoxaAI/datasets/hips_800_sort_4.h5'


def run_test():
    dm = H5DataModule(
        data_file=data_file,
        train_folds=[0, 1, 2],
        val_fold=[4],
        test_fold=[3],
        batch_size=2,  # Small batch size for testing
        stack_channels=True,  # Set to True if you want to stack channels
        target_var='target',  # Can be 'target' or 'diagnosis' depending on your dataset
    )

    dm.setup(stage='fit')

    # Iterate through training data
    print("Training Data:")
    for batch_idx, (x, y) in enumerate(dm.train_dataloader()):
        print(f"Batch {batch_idx}: x shape {x.shape}, y shape {y.shape}")
        if batch_idx == 1:  # Check a couple of batches and then break
            break

    # Optionally, test validation and test data loaders similarly
    print("\nValidation Data:")
    for batch_idx, (x, y) in enumerate(dm.val_dataloader()):
        print(f"Batch {batch_idx}: x shape {x.shape}, y shape {y.shape}")
        if batch_idx == 1:
            break

    print("\nTest Data:")
    for batch_idx, (x, y) in enumerate(dm.test_dataloader()):
        print(f"Batch {batch_idx}: x shape {x.shape}, y shape {y.shape}")
        if batch_idx == 1:
            break


if __name__ == "__main__":
    run_test()
