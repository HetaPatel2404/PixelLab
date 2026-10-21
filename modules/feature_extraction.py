
import cv2
import numpy as np


def to_grayscale(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def extract_orb_features(image, max_features=500):
    gray = to_grayscale(image)

    orb = cv2.ORB_create(
        nfeatures=max(1, int(max_features))
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray, None
    )

    result = cv2.drawKeypoints(
        gray,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return result, keypoints, descriptors


def calculate_hog(image):
    gray = to_grayscale(image)

    resized = cv2.resize(gray, (128, 128))

    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    descriptor = hog.compute(resized)

    return descriptor.flatten()


def match_orb_features(image1, image2, max_features=500):
    gray1 = to_grayscale(image1)
    gray2 = to_grayscale(image2)

    orb = cv2.ORB_create(
        nfeatures=max(1, int(max_features))
    )

    keypoints1, descriptors1 = orb.detectAndCompute(
        gray1, None
    )
    keypoints2, descriptors2 = orb.detectAndCompute(
        gray2, None
    )

    if descriptors1 is None or descriptors2 is None:
        return None, []

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING, crossCheck=True
    )

    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda match: match.distance)

    matched_image = cv2.drawMatches(
        gray1, keypoints1,
        gray2, keypoints2,
        matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return matched_image, matches
