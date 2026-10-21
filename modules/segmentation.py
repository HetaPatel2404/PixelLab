
import cv2
import numpy as np


def to_grayscale(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def binary_threshold(image, threshold=127):
    gray = to_grayscale(image)

    _, result = cv2.threshold(
        gray, threshold, 255, cv2.THRESH_BINARY
    )

    return result


def otsu_threshold(image):
    gray = to_grayscale(image)

    _, result = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


def adaptive_threshold(image, block_size=11, constant_c=2):
    gray = to_grayscale(image)

    block_size = max(3, int(block_size))

    if block_size % 2 == 0:
        block_size += 1

    if min(gray.shape) <= block_size:
        raise ValueError(
            "Block size must be smaller than image dimensions"
        )

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant_c
    )


def watershed_segmentation(image):
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image.copy()

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=2
    )

    sure_background = cv2.dilate(
        opening, kernel, iterations=3
    )

    distance = cv2.distanceTransform(
        opening, cv2.DIST_L2, 5
    )

    if distance.max() == 0:
        return rgb.copy(), np.zeros_like(gray)

    _, sure_foreground = cv2.threshold(
        distance,
        0.4 * distance.max(),
        255,
        cv2.THRESH_BINARY
    )

    sure_foreground = sure_foreground.astype(np.uint8)

    unknown = cv2.subtract(
        sure_background, sure_foreground
    )

    _, markers = cv2.connectedComponents(sure_foreground)

    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(bgr, markers)

    result = rgb.copy()
    result[markers == -1] = [255, 0, 0]

    return result, markers
