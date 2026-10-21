
import cv2
import numpy as np


def prepare_image_for_encoding(image, color_order="RGB"):
    """Convert an RGB image to OpenCV's BGR format before encoding."""
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 3:
        if color_order.upper() == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    if image.ndim == 3 and image.shape[2] == 4:
        if color_order.upper() == "RGBA":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

        return image

    raise ValueError("Unsupported image shape")


def encode_for_download(image, image_format="PNG", color_order="RGB"):
    formats = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "WEBP": ".webp"
    }

    image_format = image_format.upper()
    extension = formats.get(image_format)

    if extension is None:
        raise ValueError("Choose PNG, JPEG, or WEBP")

    image_to_encode = prepare_image_for_encoding(
        image, color_order
    )

    if image_format in ("JPEG", "JPG"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif image_format == "WEBP":
        params = [cv2.IMWRITE_WEBP_QUALITY, 95]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    success, encoded = cv2.imencode(
        extension, image_to_encode, params
    )

    if not success:
        raise RuntimeError("Unable to encode the image")

    return encoded.tobytes()


def make_download_filename(name, image_format="PNG"):
    extensions = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "WEBP": ".webp"
    }

    extension = extensions.get(image_format.upper())

    if extension is None:
        raise ValueError("Unsupported image format")

    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in name.strip()
    ).strip("_")

    if not safe_name:
        safe_name = "pixellab_processed"

    return safe_name + extension
