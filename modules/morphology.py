
import cv2
import numpy as np


def prepare_binary(image, threshold=127):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(
        image, threshold, 255, cv2.THRESH_BINARY
    )

    return binary


def make_kernel(kernel_size=3, shape="rectangle"):
    kernel_size = max(1, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    shapes = {
        "rectangle": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }

    return cv2.getStructuringElement(
        shapes.get(shape, cv2.MORPH_RECT),
        (kernel_size, kernel_size)
    )


def dilate_image(image, kernel_size=3, shape="rectangle"):
    binary = prepare_binary(image)
    kernel = make_kernel(kernel_size, shape)
    return cv2.dilate(binary, kernel, iterations=1)


def erode_image(image, kernel_size=3, shape="rectangle"):
    binary = prepare_binary(image)
    kernel = make_kernel(kernel_size, shape)
    return cv2.erode(binary, kernel, iterations=1)


def opening(image, kernel_size=3, shape="rectangle"):
    binary = prepare_binary(image)
    kernel = make_kernel(kernel_size, shape)
    return cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel
    )


def closing(image, kernel_size=3, shape="rectangle"):
    binary = prepare_binary(image)
    kernel = make_kernel(kernel_size, shape)
    return cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )


def morphological_gradient(image, kernel_size=3):
    binary = prepare_binary(image)
    kernel = make_kernel(kernel_size)
    return cv2.morphologyEx(
        binary, cv2.MORPH_GRADIENT, kernel
    )


def extract_boundary(image, kernel_size=3):
    binary = prepare_binary(image)
    kernel = make_kernel(kernel_size)
    eroded = cv2.erode(binary, kernel, iterations=1)

    return cv2.subtract(binary, eroded)


def count_connected_components(image, threshold=127):
    binary = prepare_binary(image, threshold)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    )

    objects = []

    # Label 0 represents the background, so skip it.
    for label in range(1, count):
        x, y, width, height, area = stats[label]

        objects.append({
            "label": int(label),
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "area": int(area),
            "centroid": centroids[label].tolist()
        })

    return len(objects), labels, objects
