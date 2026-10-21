
import cv2
import numpy as np


def mean_filter(image, kernel_size=5):
    kernel_size = max(1, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.blur(image, (kernel_size, kernel_size))


def gaussian_filter(image, kernel_size=5, sigma=0):
    kernel_size = max(1, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigma
    )


def median_filter(image, kernel_size=5):
    kernel_size = max(3, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.medianBlur(image, kernel_size)


def add_gaussian_noise(image, mean=0, sigma=20):
    rng = np.random.default_rng()

    noise = rng.normal(
        mean, sigma, image.shape
    ).astype(np.float32)

    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.03):
    if not 0 <= amount <= 1:
        raise ValueError("Amount must be between 0 and 1")

    result = image.copy()
    rng = np.random.default_rng()

    height, width = image.shape[:2]
    count = int(amount * height * width)

    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)

    salt_count = count // 2

    if image.ndim == 2:
        result[ys[:salt_count], xs[:salt_count]] = 255
        result[ys[salt_count:], xs[salt_count:]] = 0
    else:
        result[ys[:salt_count], xs[:salt_count], :] = 255
        result[ys[salt_count:], xs[salt_count:], :] = 0

    return result


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(
        image, diameter, sigma_color, sigma_space
    )


def arithmetic_mean_filter(image, kernel_size=3):
    return mean_filter(image, kernel_size)


def geometric_mean_filter(image, kernel_size=3):
    kernel_size = max(1, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    image_float = image.astype(np.float32) + 1.0
    log_image = np.log(image_float)

    mean_log = cv2.blur(
        log_image, (kernel_size, kernel_size)
    )

    result = np.exp(mean_log) - 1.0
    return np.clip(result, 0, 255).astype(np.uint8)


def wiener_filter(image, kernel_size=5, noise_variance=None):
    """Local adaptive Wiener-style denoising."""
    kernel_size = max(3, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    image_float = image.astype(np.float32)

    local_mean = cv2.blur(
        image_float, (kernel_size, kernel_size)
    )
    local_mean_square = cv2.blur(
        image_float ** 2, (kernel_size, kernel_size)
    )

    local_variance = np.maximum(
        local_mean_square - local_mean ** 2, 0
    )

    if noise_variance is None:
        noise_variance = float(np.mean(local_variance)) * 0.1

    noise_variance = max(0.0, float(noise_variance))

    gain = np.maximum(
        local_variance - noise_variance, 0
    ) / (local_variance + 1e-8)

    result = local_mean + gain * (image_float - local_mean)

    return np.clip(result, 0, 255).astype(np.uint8)
