import h5py
import numpy as np
import os
import dotenv
import argparse
dotenv.load_dotenv()


def explore_hdf5_group(group, prefix=''):
    """
    Recursively explores and prints information about an HDF5 group and its contents.

    :param group: h5py Group object to explore.
    :param prefix: String prefix for hierarchical display.
    """
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):
            # Print dataset information
            print(f"Dataset: {path}")
            print(f"  Shape: {item.shape}, Dtype: {item.dtype}")
            print_dataset_attributes(item)
            if key in ['diagnosis', 'target']:
                print_unique_values(item, key)
        elif isinstance(item, h5py.Group):
            # Print group information and explore recursively
            print(f"Group: {path}")
            print_group_attributes(item)
            explore_hdf5_group(item, prefix=path)


def print_group_attributes(group):
    """
    Prints attributes of an HDF5 group.

    :param group: h5py Group object.
    """
    for attr in group.attrs:
        print(f"  Attribute - {attr}: {group.attrs[attr]}")


def print_dataset_attributes(dataset):
    """
    Prints attributes of an HDF5 dataset.

    :param dataset: h5py Dataset object.
    """
    for attr in dataset.attrs:
        print(f"  Attribute - {attr}: {dataset.attrs[attr]}")


def print_unique_values(dataset, dataset_name):
    """
    Prints unique values of a specified dataset.

    :param dataset: h5py Dataset object.
    :param dataset_name: Name of the dataset (e.g., 'diagnosis', 'target').
    """
    data = dataset[:]
    unique_values = np.unique(data)
    print(f"  Unique values in {dataset_name}: {unique_values}")


def main(hdf5_path):
    """
    Main function to open and explore an HDF5 file.

    :param hdf5_path: Path to the HDF5 file.
    """
    try:
        with h5py.File(hdf5_path, 'r') as file:
            print(f"Exploring HDF5 file: {hdf5_path}")
            explore_hdf5_group(file)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print out structure of H5 dataset.") # noqa
    parser.add_argument("--data_file",
                        type=str,
                        default=os.getenv("DATA_FILE"),
                        help="Path to the data file. Defaults to DATA_FILE environment variable.")
    args = parser.parse_args()

    if not args.data_file:
        raise ValueError("No data file provided and DATA_FILE environment variable is not set.")

    main(args.data_file)
