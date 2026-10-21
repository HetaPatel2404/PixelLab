
import cv2
import numpy as np


def encode_image(image, image_format="JPEG", quality=85):
    formats = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp"
    }

    extension = formats.get(image_format.upper())

    if extension is None:
        raise ValueError("Supported formats: JPEG, PNG, WEBP")

    quality = int(np.clip(quality, 0, 100))

    if image_format.upper() == "PNG":
        params = [
            cv2.IMWRITE_PNG_COMPRESSION,
            int(round((100 - quality) * 9 / 100))
        ]
    elif image_format.upper() == "WEBP":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    success, encoded = cv2.imencode(
        extension, image, params
    )

    if not success:
        raise RuntimeError("Image encoding failed")

    return encoded.tobytes()


def decode_image(encoded_bytes):
    buffer = np.frombuffer(encoded_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

    if decoded is None:
        raise ValueError("Unable to decode image")

    if decoded.ndim == 3 and decoded.shape[2] == 3:
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if decoded.ndim == 3 and decoded.shape[2] == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)

    return decoded


def compression_ratio(original_size, compressed_size):
    if compressed_size <= 0:
        raise ValueError("Compressed size must be positive")

    return float(original_size / compressed_size)


def compression_savings(original_size, compressed_size):
    if original_size <= 0:
        raise ValueError("Original size must be positive")

    return float(
        (original_size - compressed_size) / original_size * 100
    )


def calculate_psnr(original, compressed):
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions")

    original_float = original.astype(np.float32)
    compressed_float = compressed.astype(np.float32)

    mse = np.mean((original_float - compressed_float) ** 2)

    if mse == 0:
        return float("inf")

    return float(10 * np.log10((255.0 ** 2) / mse))


def calculate_ssim(original, compressed):
    """Calculate SSIM using OpenCV's Gaussian-window formulation."""
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions")

    a = original.astype(np.float64)
    b = compressed.astype(np.float64)

    if a.ndim == 3:
        scores = [
            _ssim_channel(a[:, :, i], b[:, :, i])
            for i in range(a.shape[2])
        ]
        return float(np.mean(scores))

    return _ssim_channel(a, b)


def _ssim_channel(a, b):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (
        (mu_a_sq + mu_b_sq + c1) *
        (sigma_a_sq + sigma_b_sq + c2)
    )

    return float(np.mean(numerator / (denominator + 1e-12)))
