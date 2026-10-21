
import cv2
import numpy as np


def to_grayscale(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def calculate_fft(image):
    gray = to_grayscale(image).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = 20 * np.log(np.abs(spectrum) + 1)
    return magnitude


def create_frequency_mask(shape, cutoff=30, filter_type="low", order=2):
    height, width = shape[:2]
    y, x = np.ogrid[:height, :width]

    center_y, center_x = height // 2, width // 2
    distance = np.sqrt(
        (x - center_x) ** 2 + (y - center_y) ** 2
    )

    cutoff = max(float(cutoff), 1.0)

    if filter_type == "gaussian_low":
        mask = np.exp(-(distance ** 2) / (2 * cutoff ** 2))

    elif filter_type == "gaussian_high":
        mask = 1 - np.exp(-(distance ** 2) / (2 * cutoff ** 2))

    elif filter_type == "butterworth_low":
        mask = 1 / (
            1 + (distance / cutoff) ** (2 * order)
        )

    elif filter_type == "butterworth_high":
        mask = 1 / (
            1 + (cutoff / (distance + 1e-8)) ** (2 * order)
        )
        mask[center_y, center_x] = 0

    else:
        raise ValueError("Unsupported frequency filter")

    return mask.astype(np.float32)


def apply_frequency_filter(
    image,
    cutoff=30,
    filter_type="gaussian_low",
    order=2
):
    gray = to_grayscale(image).astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(gray))

    mask = create_frequency_mask(
        gray.shape, cutoff, filter_type, order
    )

    filtered_spectrum = spectrum * mask

    result = np.fft.ifft2(
        np.fft.ifftshift(filtered_spectrum)
    )

    result = np.abs(result)

    return np.clip(result, 0, 255).astype(np.uint8)
