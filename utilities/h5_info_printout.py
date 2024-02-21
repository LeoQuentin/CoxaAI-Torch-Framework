import h5py


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


hdf5_path = '/mnt/project/ngoc/CoxaAI/datasets/hips_800_sort_4.h5'
main(hdf5_path)
