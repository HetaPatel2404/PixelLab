
import cv2
import numpy as np


def to_grayscale(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def sobel_edges(image, kernel_size=3):
    gray = to_grayscale(image)

    sobel_x = cv2.Sobel(
        gray, cv2.CV_64F, 1, 0, ksize=kernel_size
    )
    sobel_y = cv2.Sobel(
        gray, cv2.CV_64F, 0, 1, ksize=kernel_size
    )

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return cv2.convertScaleAbs(magnitude)


def canny_edges(image, low_threshold=100, high_threshold=200):
    if low_threshold >= high_threshold:
        raise ValueError(
            "High threshold must exceed low threshold"
        )

    gray = to_grayscale(image)

    return cv2.Canny(
        gray, low_threshold, high_threshold
    )


def prewitt_edges(image):
    gray = to_grayscale(image).astype(np.float32)

    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32)

    edge_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    edge_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

    magnitude = np.sqrt(edge_x ** 2 + edge_y ** 2)

    return cv2.convertScaleAbs(magnitude)


def laplacian_edges(image, kernel_size=3):
    gray = to_grayscale(image)

    result = cv2.Laplacian(
        gray, cv2.CV_64F, ksize=kernel_size
    )

    return cv2.convertScaleAbs(result)
