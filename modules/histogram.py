
import cv2
import numpy as np


def calculate_histogram(image):
    """Return histogram data for grayscale or RGB images."""
    if image.ndim == 2:
        return {
            "Grayscale": cv2.calcHist(
                [image], [0], None, [256], [0, 256]
            ).flatten()
        }

    return {
        "Red": cv2.calcHist(
            [image], [0], None, [256], [0, 256]
        ).flatten(),
        "Green": cv2.calcHist(
            [image], [1], None, [256], [0, 256]
        ).flatten(),
        "Blue": cv2.calcHist(
            [image], [2], None, [256], [0, 256]
        ).flatten(),
    }


def equalize_grayscale(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return cv2.equalizeHist(image)


def equalize_color_image(image):
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    equalized = cv2.equalizeHist(lightness)
    merged = cv2.merge([equalized, channel_a, channel_b])

    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def histogram_statistics(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    return {
        "Mean Intensity": float(np.mean(gray)),
        "Standard Deviation": float(np.std(gray)),
        "Minimum Intensity": int(np.min(gray)),
        "Maximum Intensity": int(np.max(gray)),
        "Median Intensity": float(np.median(gray)),
    }


def histogram_similarity(image_a, image_b):
    """Compare two grayscale or RGB images using normalized histograms."""
    def make_gray(image):
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    gray_a = make_gray(image_a)
    gray_b = make_gray(image_b)

    hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [256], [0, 256])

    cv2.normalize(hist_a, hist_a, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(hist_b, hist_b, alpha=1, norm_type=cv2.NORM_L1)

    return float(cv2.compareHist(
        hist_a, hist_b, cv2.HISTCMP_CORREL
    ))
