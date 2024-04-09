import torchvision.transforms as transforms
from coxaaitorch.augmentation import ImageNetPolicy
import logging
import functools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optional_preprocessor(augmentation_func):
    """
    A decorator to apply a preprocessor (hugging face) to the output of an augmentation function.
    """
    @functools.wraps(augmentation_func)
    def wrapper(image, size=(800, 800), channels=1, preprocessor=None):
        # Apply the augmentation function
        image = augmentation_func(image, size, channels)

        # Apply the preprocessor if provided
        if preprocessor is not None:
            try:
                image = preprocessor(
                    images=image,
                    return_tensors="pt",
                    input_data_format="channels_first",
                    do_rescale=False,
                )
                image = image["pixel_values"]
                if len(image.size()) == 4:
                    image = image.squeeze(0)

                # Check if the size of the augmented image matches the preprocessor's expected size
                if image.shape[-1].item() != size:
                    logger.warning(
                        f"Mismatched sizes: Augmentation size {size} does not match preprocessor size {image.shape[-2:].tolist()}" # noqa
                    )
            except Exception as e:
                logger.error(f"Error occurred during preprocessing: {str(e)}")
                raise

        return image

    return wrapper


@optional_preprocessor
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


@optional_preprocessor
def light_augmentation(image, size=(800, 800), channels=1):
    # image is a numpy array in the shape (H, W, C)

    # Preprocess the image
    transform_pipeline = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=channels),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)
    return image


@optional_preprocessor
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


@optional_preprocessor
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
