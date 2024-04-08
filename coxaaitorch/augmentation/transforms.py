import torchvision.transforms as transforms
from coxaaitorch.augmentation import ImageNetPolicy

# import kornia.augmentation as K


def no_augmentation(image, size=(800, 800), channels=1):
    # basically same as train_preprocess but without the augmentations

    transform_pipeline = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=channels),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)
    return image


def light_augmentation(image, size=(800, 800), channels=1):
    # image is a numpy array in the shape (H, W, C)

    # Preprocess the image
    transform_pipeline = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)
    return image


def autoaugment_policy_augmentation(image, size=(800, 800), channels=1):
    # image is a numpy array in the shape (H, W, C)

    # Preprocess the image
    transform_pipeline = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            ImageNetPolicy(),
            transforms.Grayscale(num_output_channels=channels),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)
    return image


def random_augmentation(image, size=(800, 800), channels=1):
    # image is a numpy array in the shape (H, W, C)

    # Preprocess the image
    transform_pipeline = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.RandAugment(),
            transforms.Grayscale(num_output_channels=channels),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)
    return image
