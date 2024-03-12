from PIL import Image
import numpy as np


def np_image_to_PIL(image, input_range='0_1', color_conversion=None):
    """ # noqa
    Converts a NumPy array to a PIL Image, ensuring the array is in the correct format and optionally adjusting the color space.

    Args:
        image (np.ndarray): The image array to convert. Expected to have shape (H, W) or (H, W, C).
        input_range (str, optional): The range of input image values. Defaults to '0_1'. Other options: '0_255', '-1_1'.
        color_conversion (str, optional): If set, converts the color space from the specified format to RGB. 
            For example, 'BGR' to convert from BGR to RGB format.

    Returns:
        PIL.Image: The converted PIL Image, or None if conversion fails.
    """
    # Adjust the image based on the input range
    if input_range == '0_1':
        image = (image * 255).astype(np.uint8)
    elif input_range == '-1_1':
        image = ((image + 1) * 127.5).astype(np.uint8)
    elif input_range != '0_255':
        image = image.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported input range: {input_range}")

    # Convert color space if necessary
    if color_conversion == 'BGR':
        image = image[..., ::-1]

    # Remove the channel dimension if it's 1
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)

    # Convert the numpy array to a PIL Image
    try:
        image = Image.fromarray(image)
    except TypeError as e:
        print(f"Error converting array to image: {e}")
        return None

    return image
