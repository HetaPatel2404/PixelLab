
import cv2
import numpy as np


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    result = image.astype(np.float32) * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def negative_image(image):
    return 255 - image


def contrast_stretching(image):
    image_float = image.astype(np.float32)
    min_value = np.min(image_float)
    max_value = np.max(image_float)

    if max_value == min_value:
        return image.copy()

    result = (image_float - min_value) * (
        255.0 / (max_value - min_value)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def power_law_transform(image, gamma=1.0):
    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero")

    normalized = image.astype(np.float32) / 255.0
    result = np.power(normalized, gamma) * 255.0

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=8):
    if image.ndim == 2:
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size)
        )
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )

    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge(
        [enhanced_lightness, channel_a, channel_b]
    )

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def sharpen_image(image, strength=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    result = cv2.addWeighted(
        image, 1.0 + strength,
        blurred, -strength,
        0
    )
    return result


def laplacian_sharpen(image):
    if image.ndim == 2:
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        result = image.astype(np.float32) - laplacian
        return np.clip(result, 0, 255).astype(np.uint8)

    result = np.zeros_like(image)

    for channel in range(image.shape[2]):
        laplacian = cv2.Laplacian(
            image[:, :, channel], cv2.CV_32F
        )
        sharpened = image[:, :, channel].astype(np.float32) - laplacian
        result[:, :, channel] = np.clip(
            sharpened, 0, 255
        ).astype(np.uint8)

    return result
