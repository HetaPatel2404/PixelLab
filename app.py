
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from modules.learning_content import (
    render_learning_section,
    render_learning_hub
)

from modules.image_fundamentals import (
    get_image_information,
    convert_color_space,
    convert_for_display,
    quantize_image,
    resize_image,
)

from modules.enhancement import (
    adjust_brightness_contrast,
    negative_image,
    contrast_stretching,
    power_law_transform,
    apply_clahe,
    sharpen_image,
    laplacian_sharpen,
)

from modules.histogram import (
    calculate_histogram,
    equalize_grayscale,
    equalize_color_image,
    histogram_statistics,
    histogram_similarity,
)

from modules.filters import (
    mean_filter,
    gaussian_filter,
    median_filter,
    add_gaussian_noise,
    add_salt_pepper_noise,
    bilateral_filter,
    arithmetic_mean_filter,
    geometric_mean_filter,
    wiener_filter,
)

from modules.edge_detection import (
    sobel_edges,
    canny_edges,
    prewitt_edges,
    laplacian_edges,
)

from modules.frequency_domain import (
    calculate_fft,
    apply_frequency_filter,
)

from modules.morphology import (
    prepare_binary,
    dilate_image,
    erode_image,
    opening,
    closing,
    morphological_gradient,
    extract_boundary,
    count_connected_components,
)

from modules.segmentation import (
    binary_threshold,
    otsu_threshold,
    adaptive_threshold,
    watershed_segmentation,
)

from modules.feature_extraction import (
    extract_orb_features,
    match_orb_features,
)

from modules.compression import (
    decode_image,
    compression_ratio,
    compression_savings,
    calculate_psnr,
    calculate_ssim,
)

from modules.downloads import (
    encode_for_download,
    make_download_filename,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PixelLab | Digital Image Processing Studio",
    page_icon="🧪",
    layout="wide",
)


# =========================================================
# PIXELLAB PASTEL UI THEME
# =========================================================

from pathlib import Path

css_path = Path(__file__).parent / "assets" / "style.css"

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as css_file:
        pastel_css = css_file.read()

    st.markdown(
        f"<style>{pastel_css}</style>",
        unsafe_allow_html=True
    )


st.title("🧪 PixelLab")
st.markdown(
    "### Interactive Digital Image Processing Studio"
)
st.caption(
    "Explore image fundamentals, enhancement, restoration, "
    "frequency analysis, morphology, segmentation and features."
)

st.divider()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def to_gray(image):
    """Convert an RGB or grayscale image to grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def display_image(image, caption=None, width=None):
    """Safely display images with the correct numeric range."""
    if image is None:
        st.info("No output image is available.")
        return

    image = np.asarray(image)

    if image.dtype == np.bool_:
        image = image.astype(np.uint8) * 255

    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image)

        # Convert normalized float images or arbitrary-range
        # visualization outputs into a safe display range.
        if image.size and image.min() >= 0 and image.max() <= 1:
            image = image * 255
        else:
            image = np.clip(image, 0, 255)

        image = image.astype(np.uint8)

    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    
    st.image(
        image,
        caption=caption,
        use_container_width=True,
    )



def safe_visualization(image):
    """Normalize an image for visualization without division by zero."""
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image)

    minimum = float(image.min())
    maximum = float(image.max())

    if maximum <= minimum:
        return np.zeros(image.shape, dtype=np.uint8)

    result = (image - minimum) / (maximum - minimum)
    return (result * 255).astype(np.uint8)


def make_histogram_plot(image):
    """Create a histogram figure from an RGB or grayscale image."""
    fig, ax = plt.subplots(figsize=(8, 3.5))

    if image.ndim == 2:
        ax.hist(
            image.ravel(),
            bins=256,
            range=(0, 256),
            color="gray",
        )
        ax.set_title("Grayscale Intensity Histogram")
    else:
        colors = [
            ("Red", 0, "red"),
            ("Green", 1, "green"),
            ("Blue", 2, "blue"),
        ]

        for name, channel, color in colors:
            ax.hist(
                image[:, :, channel].ravel(),
                bins=256,
                range=(0, 256),
                color=color,
                alpha=0.4,
                label=name,
            )

        ax.set_title("RGB Histogram")
        ax.legend()

    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def custom_hog_descriptor(image, bins=9):
    """
    Lightweight HOG-style descriptor that does not depend on
    cv2.HOGDescriptor being available.
    """
    gray = to_gray(image)
    gray = cv2.resize(gray, (128, 128))

    gray = gray.astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude, angle = cv2.cartToPolar(
        gx,
        gy,
        angleInDegrees=True,
    )

    angle = np.mod(angle, 180)

    cell_size = 8
    descriptor_parts = []

    for y in range(0, 128, cell_size):
        for x in range(0, 128, cell_size):
            cell_mag = magnitude[
                y:y + cell_size,
                x:x + cell_size,
            ]
            cell_ang = angle[
                y:y + cell_size,
                x:x + cell_size,
            ]

            hist, _ = np.histogram(
                cell_ang,
                bins=bins,
                range=(0, 180),
                weights=cell_mag,
            )

            descriptor_parts.extend(hist.tolist())

    descriptor = np.asarray(
        descriptor_parts,
        dtype=np.float32,
    )

    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm

    return descriptor


def show_image_comparison(original, processed, labels=None):
    """Show original and processed images side by side."""
    labels = labels or ("Original Image", "Processed Image")

    col1, col2 = st.columns(2)

    with col1:
        display_image(original, labels[0])

    with col2:
        display_image(processed, labels[1])


def safe_stat_value(stats, *possible_keys):
    """Read statistics using whichever supported key is present."""
    for key in possible_keys:
        if key in stats:
            return stats[key]
    return None


# =========================================================
# SIDEBAR AND IMAGE UPLOAD
# =========================================================

st.sidebar.title("🧭 PixelLab Navigation")

page = st.sidebar.selectbox(
    "Choose a module",
    [
        "Overview",
        "Image Fundamentals",
        "Enhancement",
        "Histogram Analysis",
        "Filters & Image Restoration",
        "Frequency Domain Processing",
        "Edge Detection",
        "Morphological Operations",
        "Image Segmentation",
        "Feature Extraction",
        "Compression & Quality",
    ],
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is None:
    st.info(
        "👈 Upload an image from the sidebar to start processing."
    )

    st.markdown("### What you can explore")

    overview_cols = st.columns(3)

    cards = [
        ("🖼️", "Image Fundamentals", "Color spaces, image information, sampling and quantization."),
        ("✨", "Enhancement", "Brightness, contrast, gamma and sharpening."),
        ("📊", "Histogram Analysis", "Intensity distributions, statistics and equalization."),
        ("🧹", "Restoration", "Noise generation, smoothing and denoising."),
        ("〰️", "Frequency Domain", "FFT spectrum and frequency filters."),
        ("🧩", "Morphology", "Dilation, erosion, opening, closing and boundaries."),
        ("🎯", "Segmentation", "Thresholding, Otsu, adaptive thresholding and watershed."),
        ("🔎", "Feature Extraction", "ORB keypoints and HOG-style descriptors."),
        ("📦", "Compression", "Compare formats, file sizes and image quality."),
    ]

    for index, (emoji, title, description) in enumerate(cards):
        with overview_cols[index % 3]:
            with st.container(border=True):
                st.markdown(f"#### {emoji} {title}")
                st.write(description)

    st.stop()


try:
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_image = np.array(pil_image)
except Exception as error:
    st.error(f"Could not open the uploaded image: {error}")
    st.stop()


height, width = original_image.shape[:2]

st.sidebar.success("Image uploaded successfully")
st.sidebar.caption(f"Dimensions: {width} × {height}")

processed_image = original_image.copy()
download_label = "Download Processed Image"
download_format = "PNG"


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":
    st.header("🏠 Image Overview")

    col1, col2 = st.columns(2)

    with col1:
        display_image(original_image, "Original Image")

    with col2:
        gray = to_gray(original_image)
        display_image(gray, "Grayscale Image")

    st.subheader("Image Information")

    info = get_image_information(original_image)

    info_cols = st.columns(4)

    info_cols[0].metric("Width", f"{width} px")
    info_cols[1].metric("Height", f"{height} px")
    info_cols[2].metric("Total Pixels", f"{width * height:,}")
    info_cols[3].metric(
        "Channels",
        str(original_image.shape[2]),
    )

    st.json(info)

    st.subheader("RGB Histogram")

    fig = make_histogram_plot(original_image)
    st.pyplot(fig)
    plt.close(fig)

    render_learning_hub()

# =========================================================
# IMAGE FUNDAMENTALS
# =========================================================

elif page == "Image Fundamentals":
    st.header("🖼️ Image Fundamentals")

    operation = st.selectbox(
        "Choose an operation",
        [
            "Image Information",
            "Color Space Conversion",
            "Negative Image",
            "Quantization",
            "Sampling / Resize",
        ],
    )

    if operation == "Image Information":
        st.subheader("Image Properties")

        info = get_image_information(original_image)
        st.json(info)

        st.caption(
            "Sampling determines spatial resolution, while quantization "
            "determines the number of available intensity levels."
        )

    elif operation == "Color Space Conversion":
        color_space = st.selectbox(
            "Color space",
            ["RGB", "Grayscale", "HSV", "LAB", "YCrCb"],
        )

        converted = convert_color_space(
            original_image,
            color_space,
        )

        if color_space == "RGB":
            display_result = converted
        else:
            display_result = convert_for_display(
                converted,
                color_space,
            )

        show_image_comparison(
            original_image,
            display_result,
            ("Original RGB", f"{color_space} Visualization"),
        )

        processed_image = display_result

    elif operation == "Negative Image":
        processed_image = negative_image(original_image)

        show_image_comparison(
            original_image,
            processed_image,
            ("Original Image", "Negative Image"),
        )

    elif operation == "Quantization":
        levels = st.select_slider(
            "Intensity levels",
            options=[2, 4, 8, 16, 32, 64, 128, 256],
            value=8,
        )

        processed_image = quantize_image(
            original_image,
            levels=levels,
        )

        show_image_comparison(
            original_image,
            processed_image,
            ("Original Image", "Quantized Image"),
        )

        st.caption(f"Quantization levels: {levels}")

    elif operation == "Sampling / Resize":
        scale = st.slider(
            "Scale factor",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )

        resized = resize_image(
            original_image,
            scale=scale,
        )

        col1, col2 = st.columns(2)

        with col1:
            display_image(original_image, "Original Image")

        with col2:
            display_image(resized, "Resized Image")

        st.write(
            f"Original: {width} × {height} | "
            f"Resized: {resized.shape[1]} × {resized.shape[0]}"
        )

        processed_image = resized

    render_learning_section("Image Fundamentals")

# =========================================================
# ENHANCEMENT
# =========================================================

elif page == "Enhancement":
    st.header("✨ Image Enhancement")

    operation = st.selectbox(
        "Choose an operation",
        [
            "Brightness & Contrast",
            "Negative Image",
            "Contrast Stretching",
            "Gamma Correction",
            "CLAHE",
            "Unsharp Mask",
            "Laplacian Sharpening",
        ],
    )

    if operation == "Brightness & Contrast":
        brightness = st.slider(
            "Brightness",
            -100,
            100,
            0,
        )

        contrast = st.slider(
            "Contrast",
            0.1,
            3.0,
            1.0,
            step=0.1,
        )

        processed_image = adjust_brightness_contrast(
            original_image,
            brightness=brightness,
            contrast=contrast,
        )

    elif operation == "Negative Image":
        processed_image = negative_image(original_image)

    elif operation == "Contrast Stretching":
        processed_image = contrast_stretching(original_image)

    elif operation == "Gamma Correction":
        gamma = st.slider(
            "Gamma",
            0.1,
            3.0,
            1.0,
            step=0.1,
        )

        processed_image = power_law_transform(
            original_image,
            gamma=gamma,
        )

    elif operation == "CLAHE":
        clip_limit = st.slider(
            "Clip Limit",
            1.0,
            5.0,
            2.0,
            step=0.5,
        )

        tile_size = st.slider(
            "Tile Grid Size",
            4,
            16,
            8,
        )

        processed_image = apply_clahe(
            original_image,
            clip_limit=clip_limit,
            tile_grid_size=tile_size,
        )

    elif operation == "Unsharp Mask":
        strength = st.slider(
            "Sharpening Strength",
            0.1,
            3.0,
            1.0,
            step=0.1,
        )

        processed_image = sharpen_image(
            original_image,
            strength=strength,
        )

    elif operation == "Laplacian Sharpening":
        processed_image = laplacian_sharpen(original_image)

    show_image_comparison(
        original_image,
        processed_image,
        ("Original Image", operation),
    )

    render_learning_section("Enhancement")

# =========================================================
# HISTOGRAM ANALYSIS
# =========================================================

elif page == "Histogram Analysis":
    st.header("📊 Histogram Analysis")

    operation = st.selectbox(
        "Choose an operation",
        [
            "RGB Histogram",
            "Grayscale Histogram",
            "Histogram Equalization",
            "Histogram Statistics",
            "Compare Two Images",
        ],
    )

    if operation == "RGB Histogram":
        fig = make_histogram_plot(original_image)
        st.pyplot(fig)
        plt.close(fig)

    elif operation == "Grayscale Histogram":
        gray = to_gray(original_image)

        fig = make_histogram_plot(gray)
        st.pyplot(fig)
        plt.close(fig)

        processed_image = gray

    elif operation == "Histogram Equalization":
        mode = st.radio(
            "Equalization mode",
            ["Grayscale", "Color (LAB L channel)"],
            horizontal=True,
        )

        if mode == "Grayscale":
            processed_image = equalize_grayscale(
                to_gray(original_image)
            )
        else:
            processed_image = equalize_color_image(original_image)

        show_image_comparison(
            original_image,
            processed_image,
            ("Original Image", "Equalized Image"),
        )

        st.subheader("Before and After Histograms")

        c1, c2 = st.columns(2)

        with c1:
            fig = make_histogram_plot(original_image)
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig = make_histogram_plot(processed_image)
            st.pyplot(fig)
            plt.close(fig)

    elif operation == "Histogram Statistics":
        stats = histogram_statistics(original_image)

        st.subheader("Grayscale Intensity Statistics")

        # Handle common key naming variations safely.
        mean_value = safe_stat_value(
            stats, "mean", "Mean", "average"
        )
        std_value = safe_stat_value(
            stats, "std", "standard_deviation", "std_dev"
        )
        min_value = safe_stat_value(
            stats, "min", "minimum"
        )
        max_value = safe_stat_value(
            stats, "max", "maximum"
        )
        median_value = safe_stat_value(
            stats, "median", "Median"
        )

        columns = st.columns(5)

        values = [
            ("Mean", mean_value),
            ("Std. Deviation", std_value),
            ("Minimum", min_value),
            ("Maximum", max_value),
            ("Median", median_value),
        ]

        for column, (label, value) in zip(columns, values):
            if value is None:
                column.metric(label, "N/A")
            elif isinstance(value, (int, float, np.number)):
                column.metric(label, f"{float(value):.2f}")
            else:
                column.metric(label, str(value))

        with st.expander("View raw statistics"):
            st.json(stats)

    elif operation == "Compare Two Images":
        second_file = st.file_uploader(
            "Upload a second image to compare",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="histogram_compare",
        )

        if second_file is not None:
            second_image = np.array(
                Image.open(second_file).convert("RGB")
            )

            similarity = histogram_similarity(
                original_image,
                second_image,
            )

            st.metric(
                "Histogram Correlation",
                f"{float(similarity):.4f}",
            )

            show_image_comparison(
                original_image,
                second_image,
                ("Image 1", "Image 2"),
            )

        render_learning_section("Histogram Analysis") 

# =========================================================
# FILTERS AND IMAGE RESTORATION
# =========================================================

elif page == "Filters & Image Restoration":
    st.header("🧹 Filters & Image Restoration")

    operation = st.selectbox(
        "Choose a filter or restoration operation",
        [
            "Mean Filter",
            "Gaussian Filter",
            "Median Filter",
            "Bilateral Filter",
            "Arithmetic Mean Filter",
            "Geometric Mean Filter",
            "Wiener-style Adaptive Filter",
            "Add Gaussian Noise",
            "Add Salt & Pepper Noise",
            "Noise + Denoising Comparison",
        ],
    )

    # Each filter has its own independent kernel-size control.
    # Selecting a different operation displays that filter's settings.
    if operation == "Mean Filter":
        kernel_size = st.select_slider(
            "Mean Filter Kernel Size",
            options=[3, 5, 7, 9, 11],
            value=3,
            key="mean_kernel",
        )

        processed_image = mean_filter(
            original_image,
            kernel_size=kernel_size,
        )

    elif operation == "Gaussian Filter":
        kernel_size = st.select_slider(
            "Gaussian Filter Kernel Size",
            options=[3, 5, 7, 9, 11],
            value=5,
            key="gaussian_kernel",
        )

        sigma = st.slider(
            "Gaussian Sigma",
            0.0,
            5.0,
            0.0,
            step=0.1,
            key="gaussian_sigma",
        )

        processed_image = gaussian_filter(
            original_image,
            kernel_size=kernel_size,
            sigma=sigma,
        )

    elif operation == "Median Filter":
        kernel_size = st.select_slider(
            "Median Filter Kernel Size",
            options=[3, 5, 7, 9, 11],
            value=3,
            key="median_kernel",
        )

        processed_image = median_filter(
            original_image,
            kernel_size=kernel_size,
        )

    elif operation == "Bilateral Filter":
        diameter = st.slider(
            "Bilateral Diameter",
            3,
            15,
            9,
            step=2,
        )

        sigma_color = st.slider(
            "Sigma Color",
            10,
            150,
            75,
        )

        sigma_space = st.slider(
            "Sigma Space",
            10,
            150,
            75,
        )

        processed_image = bilateral_filter(
            original_image,
            diameter=diameter,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
        )

    elif operation == "Arithmetic Mean Filter":
        kernel_size = st.select_slider(
            "Arithmetic Mean Window Size",
            options=[3, 5, 7, 9],
            value=3,
            key="arithmetic_kernel",
        )

        processed_image = arithmetic_mean_filter(
            original_image,
            kernel_size=kernel_size,
        )

    elif operation == "Geometric Mean Filter":
        kernel_size = st.select_slider(
            "Geometric Mean Window Size",
            options=[3, 5, 7, 9],
            value=3,
            key="geometric_kernel",
        )

        processed_image = geometric_mean_filter(
            original_image,
            kernel_size=kernel_size,
        )

    elif operation == "Wiener-style Adaptive Filter":
        kernel_size = st.select_slider(
            "Adaptive Window Size",
            options=[3, 5, 7, 9],
            value=5,
            key="wiener_kernel",
        )

        processed_image = wiener_filter(
            original_image,
            kernel_size=kernel_size,
        )

    elif operation == "Add Gaussian Noise":
        sigma = st.slider(
            "Noise Sigma",
            1,
            80,
            20,
        )

        processed_image = add_gaussian_noise(
            original_image,
            sigma=sigma,
        )

    elif operation == "Add Salt & Pepper Noise":
        amount = st.slider(
            "Noise Amount",
            0.01,
            0.15,
            0.03,
            step=0.01,
        )

        processed_image = add_salt_pepper_noise(
            original_image,
            amount=amount,
        )

    elif operation == "Noise + Denoising Comparison":
        noise_type = st.selectbox(
            "Noise Type",
            ["Gaussian", "Salt & Pepper"],
        )

        kernel_size = st.select_slider(
            "Denoising Kernel Size",
            options=[3, 5, 7, 9],
            value=5,
            key="noise_denoise_kernel",
        )

        if noise_type == "Gaussian":
            noisy = add_gaussian_noise(
                original_image,
                sigma=20,
            )
        else:
            noisy = add_salt_pepper_noise(
                original_image,
                amount=0.04,
            )

        restored = median_filter(
            noisy,
            kernel_size=kernel_size,
        )

        st.subheader("Restoration Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            display_image(original_image, "Original")

        with col2:
            display_image(noisy, "Noisy Image")

        with col3:
            display_image(restored, "Denoised Image")

        processed_image = restored

    if operation != "Noise + Denoising Comparison":
        show_image_comparison(
            original_image,
            processed_image,
            ("Original Image", operation),
        )

        render_learning_section("Filters & Image Restoration")

# =========================================================
# FREQUENCY DOMAIN PROCESSING
# =========================================================

elif page == "Frequency Domain Processing":
    st.header("〰️ Frequency Domain Processing")

    operation = st.selectbox(
        "Choose an operation",
        [
            "FFT Magnitude Spectrum",
            "Gaussian Low-Pass Filter",
            "Gaussian High-Pass Filter",
            "Butterworth Low-Pass Filter",
            "Butterworth High-Pass Filter",
        ],
    )

    gray = to_gray(original_image)

    if operation == "FFT Magnitude Spectrum":
        spectrum = calculate_fft(gray)

        # Fix: scale the FFT result before showing it with Streamlit.
        spectrum_display = safe_visualization(spectrum)

        display_image(gray, "Grayscale Input")
        display_image(
            spectrum_display,
            "FFT Magnitude Spectrum (Normalized)",
        )

        processed_image = spectrum_display

    else:
        cutoff = st.slider(
            "Frequency Cutoff",
            5,
            100,
            30,
        )

        order = st.slider(
            "Butterworth Order",
            1,
            6,
            2,
        )

        filter_mapping = {
            "Gaussian Low-Pass Filter": "gaussian_low",
            "Gaussian High-Pass Filter": "gaussian_high",
            "Butterworth Low-Pass Filter": "butterworth_low",
            "Butterworth High-Pass Filter": "butterworth_high",
        }

        filter_type = filter_mapping[operation]

        processed_image = apply_frequency_filter(
            gray,
            cutoff=cutoff,
            filter_type=filter_type,
            order=order,
        )

        show_image_comparison(
            gray,
            processed_image,
            ("Grayscale Input", operation),
        )

    render_learning_section("Frequency Domain Processing")

# =========================================================
# EDGE DETECTION
# =========================================================

elif page == "Edge Detection":
    st.header("📐 Edge Detection")

    operation = st.selectbox(
        "Choose an edge detector",
        [
            "Sobel",
            "Canny",
            "Prewitt",
            "Laplacian",
        ],
    )

    if operation == "Sobel":
        kernel_size = st.select_slider(
            "Sobel Kernel Size",
            options=[3, 5, 7],
            value=3,
        )

        processed_image = sobel_edges(
            original_image,
            kernel_size=kernel_size,
        )

    elif operation == "Canny":
        low_threshold = st.slider(
            "Low Threshold",
            0,
            250,
            100,
        )

        high_threshold = st.slider(
            "High Threshold",
            1,
            255,
            200,
        )

        if low_threshold >= high_threshold:
            st.warning(
                "High Threshold must be greater than Low Threshold."
            )
            high_threshold = min(
                255,
                low_threshold + 1,
            )

        processed_image = canny_edges(
            original_image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

    elif operation == "Prewitt":
        processed_image = prewitt_edges(original_image)

    else:
        kernel_size = st.select_slider(
            "Laplacian Kernel Size",
            options=[1, 3, 5, 7],
            value=3,
        )

        processed_image = laplacian_edges(
            original_image,
            kernel_size=kernel_size,
        )

    show_image_comparison(
        original_image,
        processed_image,
        ("Original Image", f"{operation} Edges"),
    )

    render_learning_section("Edge Detection")

# =========================================================
# MORPHOLOGICAL OPERATIONS
# =========================================================

elif page == "Morphological Operations":
    st.header("🧩 Morphological Operations")

    operation = st.selectbox(
        "Choose a morphological operation",
        [
            "Dilation",
            "Erosion",
            "Opening",
            "Closing",
            "Morphological Gradient",
            "Boundary Extraction",
            "Connected Components",
        ],
    )

    threshold = st.slider(
        "Binary Threshold",
        0,
        255,
        127,
    )

    kernel_size = st.select_slider(
        "Structuring Element Size",
        options=[3, 5, 7, 9],
        value=3,
    )

    kernel_shape = st.selectbox(
        "Structuring Element Shape",
        ["rectangle", "ellipse", "cross"],
    )

    # Fix: morphology functions prepare their own binary inputs
    # and do not accept a threshold keyword argument.
    binary = prepare_binary(
        original_image,
        threshold=threshold,
    )

    if operation == "Dilation":
        processed_image = dilate_image(
            binary,
            kernel_size=kernel_size,
            shape=kernel_shape,
        )

    elif operation == "Erosion":
        processed_image = erode_image(
            binary,
            kernel_size=kernel_size,
            shape=kernel_shape,
        )

    elif operation == "Opening":
        processed_image = opening(
            binary,
            kernel_size=kernel_size,
            shape=kernel_shape,
        )

    elif operation == "Closing":
        processed_image = closing(
            binary,
            kernel_size=kernel_size,
            shape=kernel_shape,
        )

    
    elif operation == "Morphological Gradient":
        processed_image = morphological_gradient(
            binary,
            kernel_size=kernel_size,
        )

        processed_image = morphological_gradient(
            binary,
            kernel_size=kernel_size,
            
        )

    elif operation == "Boundary Extraction":
        processed_image = extract_boundary(
            binary,
            kernel_size=kernel_size,
            
        )

    else:
        count, labels, objects = count_connected_components(
            binary,
            threshold=127,
        )

        processed_image = safe_visualization(labels)

        st.metric(
            "Connected Components (excluding background)",
            count,
        )

        if objects:
            with st.expander("Component details"):
                st.write(objects)

    show_image_comparison(
        original_image,
        processed_image,
        ("Original Image", operation),
    )

    render_learning_section("Morphological Operations")

# =========================================================
# IMAGE SEGMENTATION
# =========================================================

elif page == "Image Segmentation":
    st.header("🎯 Image Segmentation")

    operation = st.selectbox(
        "Choose a segmentation method",
        [
            "Binary Threshold",
            "Otsu Threshold",
            "Adaptive Threshold",
            "Watershed",
        ],
    )

    gray = to_gray(original_image)

    if operation == "Binary Threshold":
        threshold = st.slider(
            "Threshold",
            0,
            255,
            127,
        )

        processed_image = binary_threshold(
            gray,
            threshold=threshold,
        )

    elif operation == "Otsu Threshold":
        processed_image = otsu_threshold(gray)

    elif operation == "Adaptive Threshold":
        block_size = st.select_slider(
            "Block Size",
            options=[3, 5, 7, 9, 11, 15, 21, 31],
            value=11,
        )

        constant_c = st.slider(
            "Constant C",
            -10,
            15,
            2,
        )

        if min(gray.shape) <= block_size:
            st.warning(
                "Block Size is too large for this image. "
                "Choose a smaller block size."
            )
            processed_image = gray
        else:
            processed_image = adaptive_threshold(
                gray,
                block_size=block_size,
                constant_c=constant_c,
            )

    else:
        processed_image, markers = watershed_segmentation(
            original_image
        )

        st.caption(
            "Watershed marks boundaries between segmented regions."
        )

    show_image_comparison(
        original_image,
        processed_image,
        ("Original Image", operation),
    )

    render_learning_section("Image Segmentation")

# =========================================================
# FEATURE EXTRACTION
# =========================================================

elif page == "Feature Extraction":
    st.header("🔎 Feature Extraction")

    method = st.selectbox(
        "Choose a feature method",
        [
            "HOG Descriptor",
            "ORB Keypoints",
            "ORB Feature Matching",
        ],
    )

    if method == "HOG Descriptor":
        descriptor = custom_hog_descriptor(original_image)

        st.success("HOG-style descriptor calculated successfully.")

        col1, col2 = st.columns(2)

        with col1:
            display_image(original_image, "Input Image")

        with col2:
            st.metric(
                "Descriptor Length",
                f"{len(descriptor):,}",
            )

            st.write("First 30 descriptor values")
            st.code(
                np.array2string(
                    descriptor[:30],
                    precision=4,
                )
            )

        st.caption(
            "This fallback uses image gradients and orientation "
            "histograms, so it works even when cv2.HOGDescriptor "
            "is unavailable."
        )

    elif method == "ORB Keypoints":
        max_features = st.slider(
            "Maximum Features",
            100,
            1500,
            500,
            step=100,
        )

        result, keypoints, descriptors = extract_orb_features(
            original_image,
            max_features=max_features,
        )

        display_image(
            result,
            f"ORB Keypoints: {len(keypoints)}",
        )

        st.metric("Keypoints Detected", len(keypoints))

        if descriptors is not None:
            st.metric(
                "Descriptor Count",
                len(descriptors),
            )

        processed_image = result

    else:
        second_file = st.file_uploader(
            "Upload a second image for matching",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="orb_matching",
        )

        if second_file is not None:
            second_image = np.array(
                Image.open(second_file).convert("RGB")
            )

            max_features = st.slider(
                "Maximum Features",
                100,
                1500,
                500,
                step=100,
            )

            matched_image, matches = match_orb_features(
                original_image,
                second_image,
                max_features=max_features,
            )

            if matched_image is None or len(matches) == 0:
                st.warning(
                    "No usable ORB matches were found. "
                    "Try images with more visible features."
                )
            else:
                # OpenCV drawMatches generally returns BGR output.
                matched_rgb = cv2.cvtColor(
                    matched_image,
                    cv2.COLOR_BGR2RGB,
                )

                display_image(
                    matched_rgb,
                    f"ORB Matches: {len(matches)}",
                )

                processed_image = matched_rgb

                st.metric(
                    "Number of Matches",
                    len(matches),
                )

    render_learning_section("Feature Extraction")

# =========================================================
# COMPRESSION AND QUALITY
# =========================================================

elif page == "Compression & Quality":
    st.header("📦 Image Compression & Quality")

    image_format = st.selectbox(
        "Output Format",
        ["JPEG", "PNG", "WEBP"],
    )

    quality = st.slider(
        "Quality / Compression Setting",
        10,
        100,
        85,
    )

    try:
        encoded = encode_for_download(
            original_image,
            image_format=image_format,
            color_order="RGB",
        )

        reconstructed = decode_image(encoded)

        original_bytes = len(uploaded_file.getvalue())
        compressed_bytes = len(encoded)

        col1, col2 = st.columns(2)

        with col1:
            display_image(
                original_image,
                "Original Image",
            )

        with col2:
            display_image(
                reconstructed,
                f"{image_format} Reconstruction",
            )

        st.subheader("Compression Metrics")

        metrics = st.columns(4)

        metrics[0].metric(
            "Original Size",
            f"{original_bytes / 1024:.2f} KB",
        )

        metrics[1].metric(
            "Encoded Size",
            f"{compressed_bytes / 1024:.2f} KB",
        )

        metrics[2].metric(
            "Compression Ratio",
            f"{compression_ratio(original_bytes, compressed_bytes):.2f}",
        )

        metrics[3].metric(
            "Space Saved",
            f"{compression_savings(original_bytes, compressed_bytes):.2f}%",
        )

        if reconstructed.shape == original_image.shape:
            psnr_value = calculate_psnr(
                original_image,
                reconstructed,
            )

            ssim_value = calculate_ssim(
                original_image,
                reconstructed,
            )

            quality_cols = st.columns(2)

            if np.isinf(psnr_value):
                quality_cols[0].metric("PSNR", "∞ dB")
            else:
                quality_cols[0].metric(
                    "PSNR",
                    f"{psnr_value:.2f} dB",
                )

            quality_cols[1].metric(
                "SSIM (approximation)",
                f"{ssim_value:.4f}",
            )
        else:
            st.warning(
                "Quality metrics were skipped because the decoded "
                "image dimensions do not match the original."
            )

        st.caption(
            "The metrics compare the uploaded image with the newly "
            "encoded reconstruction. File size depends on the "
            "source file and selected output settings."
        )

        processed_image = reconstructed
        download_format = image_format

    except Exception as error:
        st.error(f"Compression failed: {error}")
        processed_image = original_image

    render_learning_section("Compression & Quality")

# =========================================================
# SINGLE GLOBAL DOWNLOAD BUTTON
# =========================================================

st.divider()
st.subheader("⬇️ Export Processed Image")

try:
    download_data = encode_for_download(
        processed_image,
        image_format=download_format,
        color_order="RGB",
    )

    filename = make_download_filename(
        "pixellab_processed",
        download_format,
    )

    mime_types = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
    }

    st.download_button(
        label="⬇️ Download Processed Image",
        data=download_data,
        file_name=filename,
        mime=mime_types.get(download_format, "image/png"),
        use_container_width=True,
        key="pixellab_single_download",
    )

except Exception as error:
    st.error(f"Could not prepare download: {error}")


st.divider()
st.caption(
    "PixelLab • Built with Python, Streamlit, OpenCV, NumPy and Matplotlib"
)
