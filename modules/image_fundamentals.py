
import cv2
import numpy as np


def get_image_information(image):
    """Return basic information about an image."""
    height, width = image.shape[:2]

    if image.ndim == 2:
        channels = 1
        image_type = "Grayscale"
    else:
        channels = image.shape[2]
        image_type = "Color"

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "image_type": image_type,
        "dimensions": str(image.shape),
        "data_type": str(image.dtype),
        "total_pixels": width * height,
    }


def convert_color_space(image, color_space):
    """Convert an RGB image into a selected color space."""
    if color_space == "RGB":
        return image.copy()

    if color_space == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if color_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    if color_space == "YCrCb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    raise ValueError("Unsupported color space")


def convert_for_display(image, color_space):
    """Prepare converted images for correct display in Streamlit."""
    if color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    if color_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

    if color_space == "YCrCb":
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)

    return image


def negative_image(image):
    """Create the negative of a grayscale or color image."""
    return 255 - image


def quantize_image(image, levels=8):
    """Reduce the number of intensity levels in an image."""
    if levels < 2 or levels > 256:
        raise ValueError("Levels must be between 2 and 256")

    image_float = image.astype(np.float32)
    step = 256.0 / levels

    quantized = np.floor(image_float / step) * step
    quantized = np.clip(quantized, 0, 255)

    return quantized.astype(np.uint8)


def resize_image(image, scale=0.5):
    """Resize an image to demonstrate image sampling."""
    if scale <= 0:
        raise ValueError("Scale must be greater than zero")

    height, width = image.shape[:2]

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )
