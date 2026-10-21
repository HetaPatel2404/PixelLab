
import streamlit as st


LEARNING_CONTENT = {
    "Image Fundamentals": {
        "icon": "🖼️",
        "title": "How Does a Digital Image Work?",
        "intro": (
            "A digital image is made up of tiny squares called pixels. "
            "Each pixel stores information about brightness or colour. "
            "Together, these pixels create the image we see."
        ),
        "flow": ["Image", "Pixels", "Pixel Values", "Digital Display"],
        "concepts": [
            ("Pixel", "The smallest individual unit of a digital image."),
            ("Resolution", "The number of pixels in an image, usually expressed as width × height."),
            ("Grayscale", "Represents an image using shades from black to white."),
            ("RGB", "Combines Red, Green, and Blue values to represent colours."),
            ("Quantization", "Maps continuous or finely sampled values to a limited set of levels."),
        ],
        "fact": "A grayscale image commonly uses values from 0 (black) to 255 (white) when stored with 8 bits per pixel.",
        "question": "Which colour model uses Red, Green, and Blue channels?",
        "options": ["CMYK", "RGB", "Grayscale", "Binary"],
        "answer": "RGB",
        "why": "RGB represents colours by combining different intensities of red, green, and blue."
    },

    "Enhancement": {
        "icon": "✨",
        "title": "Make Images Clearer and More Informative",
        "intro": (
            "Image enhancement improves the visual appearance of an image. "
            "It can make details easier to see by adjusting brightness, "
            "contrast, intensity values, or sharpness."
        ),
        "flow": ["Original Image", "Enhancement Method", "Adjusted Pixels", "Improved Appearance"],
        "concepts": [
            ("Brightness", "Controls how light or dark an image appears."),
            ("Contrast", "Controls the difference between lighter and darker regions."),
            ("Negative", "Reverses intensity values to create an image negative."),
            ("Gamma Correction", "Adjusts brightness non-linearly using a power-law transformation."),
            ("CLAHE", "Improves local contrast while limiting excessive contrast amplification."),
            ("Sharpening", "Emphasises intensity changes to make edges and details more noticeable."),
        ],
        "fact": "Enhancement changes how an image looks; it does not necessarily recover information that was never captured.",
        "question": "Which adjustment increases the difference between bright and dark regions?",
        "options": ["Contrast", "Image Size", "File Name", "Colour Order"],
        "answer": "Contrast",
        "why": "Contrast controls the separation between lighter and darker intensity values."
    },

    "Histogram Analysis": {
        "icon": "📊",
        "title": "Read the Story Behind Pixel Intensities",
        "intro": (
            "An image histogram shows how frequently different intensity values "
            "occur. It helps us understand brightness, contrast, and the "
            "distribution of dark and bright pixels."
        ),
        "flow": ["Image Pixels", "Count Intensities", "Build Histogram", "Analyse Distribution"],
        "concepts": [
            ("Histogram", "A graph showing the frequency of each intensity value."),
            ("Intensity Range", "The span between the darkest and brightest pixel values."),
            ("Histogram Equalization", "Redistributes intensities to improve overall contrast."),
            ("Mean Intensity", "The average intensity value of the pixels."),
            ("Standard Deviation", "Measures how widely intensity values vary around their mean."),
        ],
        "fact": "A histogram concentrated on the left usually indicates darker pixels, while one concentrated on the right usually indicates brighter pixels.",
        "question": "What does an image histogram represent?",
        "options": [
            "The physical weight of an image",
            "The frequency of intensity values",
            "The image file name",
            "The number of folders"
        ],
        "answer": "The frequency of intensity values",
        "why": "A histogram counts how often each intensity value occurs in an image."
    },

    "Filters & Image Restoration": {
        "icon": "🧹",
        "title": "Clean Up Noisy Images",
        "intro": (
            "Image filtering modifies pixels using information from their "
            "neighbours. Filters can reduce noise, smooth unwanted variations, "
            "or preserve important edges and details."
        ),
        "flow": ["Noisy Image", "Neighbourhood Analysis", "Apply Filter", "Filtered Image"],
        "concepts": [
            ("Mean Filter", "Replaces a pixel using the average of nearby values."),
            ("Gaussian Filter", "Smooths an image using a weighted neighbourhood."),
            ("Median Filter", "Uses the median of neighbouring values and is useful for salt-and-pepper noise."),
            ("Bilateral Filter", "Smooths similar regions while helping preserve edges."),
            ("Gaussian Noise", "Random intensity variations that often follow a Gaussian distribution."),
            ("Wiener Filter", "Estimates a cleaner image using statistical information about the image and noise."),
        ],
        "fact": "A larger filter neighbourhood often creates stronger smoothing, but it can also remove fine details.",
        "question": "Which filter is especially useful for salt-and-pepper noise?",
        "options": ["Median Filter", "Image Negative", "Histogram Plot", "Colour Conversion"],
        "answer": "Median Filter",
        "why": "The median operation can reject isolated extreme pixel values without averaging them into surrounding pixels."
    },

    "Frequency Domain Processing": {
        "icon": "〰️",
        "title": "Explore Images as Frequencies",
        "intro": (
            "Frequency-domain processing represents an image using spatial "
            "frequencies instead of working only with its pixel positions. "
            "The Fourier Transform helps reveal broad patterns and rapid "
            "intensity changes."
        ),
        "flow": ["Spatial Image", "Fourier Transform", "Frequency Filtering", "Inverse Transform"],
        "concepts": [
            ("Fourier Transform", "Converts an image representation from the spatial domain to the frequency domain."),
            ("Low Frequencies", "Often correspond to slowly changing intensity patterns."),
            ("High Frequencies", "Often correspond to rapid changes, fine details, and edges."),
            ("Low-Pass Filter", "Reduces high-frequency components and can smooth an image."),
            ("High-Pass Filter", "Reduces low-frequency components and can emphasise details."),
        ],
        "fact": "Frequency filtering can affect many pixels at once because frequency components describe patterns across the image.",
        "question": "Which type of filter commonly reduces fine, rapid intensity variations?",
        "options": ["Low-pass filter", "High-pass filter", "Image upload", "Colour label"],
        "answer": "Low-pass filter",
        "why": "A low-pass filter attenuates higher spatial frequencies, often producing a smoothing effect."
    },

    "Edge Detection": {
        "icon": "🔎",
        "title": "Find the Boundaries That Matter",
        "intro": (
            "Edges are locations where image intensity changes noticeably. "
            "Edge detection highlights these changes and helps identify "
            "object boundaries, shapes, and structural details."
        ),
        "flow": ["Input Image", "Measure Intensity Changes", "Detect Strong Changes", "Edge Map"],
        "concepts": [
            ("Sobel Operator", "Estimates intensity gradients in horizontal and vertical directions."),
            ("Prewitt Operator", "Uses neighbourhood masks to estimate image gradients."),
            ("Laplacian Operator", "Uses second-order derivatives to detect rapid intensity changes."),
            ("Canny Detector", "Combines smoothing, gradient analysis, non-maximum suppression, and edge tracking."),
            ("Threshold", "A value used to decide which responses are strong enough to retain."),
        ],
        "fact": "Edge detection highlights boundaries, but texture and noise can also create edge responses.",
        "question": "Which edge detector uses two thresholds and edge tracking by hysteresis?",
        "options": ["Canny", "Median Filter", "Grayscale Conversion", "JPEG Encoding"],
        "answer": "Canny",
        "why": "Canny uses low and high thresholds with hysteresis to connect meaningful edge responses."
    },

    "Morphological Operations": {
        "icon": "🧩",
        "title": "Shape-Based Image Processing",
        "intro": (
            "Morphological operations analyse and modify the shapes of objects, "
            "usually in binary images. They use a small pattern called a "
            "structuring element to process foreground regions."
        ),
        "flow": ["Binary Image", "Structuring Element", "Morphological Operation", "Updated Shapes"],
        "concepts": [
            ("Dilation", "Expands foreground regions and can connect nearby objects."),
            ("Erosion", "Shrinks foreground regions and can remove small protrusions."),
            ("Opening", "Erosion followed by dilation; can remove small foreground noise."),
            ("Closing", "Dilation followed by erosion; can fill small gaps and holes."),
            ("Morphological Gradient", "Highlights boundaries using the difference between dilation and erosion."),
            ("Connected Components", "Labels separate connected foreground regions."),
        ],
        "fact": "The shape and size of the structuring element influence which image structures are preserved or removed.",
        "question": "Which operation generally expands white foreground regions?",
        "options": ["Dilation", "Erosion", "Grayscale", "Compression"],
        "answer": "Dilation",
        "why": "Dilation expands foreground regions according to the chosen structuring element."
    },

    "Image Segmentation": {
        "icon": "🎯",
        "title": "Separate Objects from Their Background",
        "intro": (
            "Image segmentation divides an image into meaningful regions. "
            "It helps separate foreground objects from backgrounds or "
            "distinguish regions with different visual properties."
        ),
        "flow": ["Input Image", "Choose Segmentation Method", "Classify Pixels", "Segmented Regions"],
        "concepts": [
            ("Thresholding", "Classifies pixels according to whether their intensity is above or below a threshold."),
            ("Otsu's Method", "Automatically chooses a threshold by analysing intensity distributions."),
            ("Adaptive Thresholding", "Calculates thresholds locally to handle varying illumination."),
            ("Watershed", "Uses markers and image structure to help separate touching regions."),
            ("Binary Mask", "An image that marks selected regions, often using foreground and background values."),
        ],
        "fact": "A single global threshold may struggle when one part of an image is bright and another is poorly illuminated.",
        "question": "Which method calculates thresholds using local neighbourhoods?",
        "options": ["Adaptive Thresholding", "Image Negative", "JPEG Compression", "ORB Matching"],
        "answer": "Adaptive Thresholding",
        "why": "Adaptive thresholding determines a threshold for each local neighbourhood rather than using one value everywhere."
    },

    "Feature Extraction": {
        "icon": "🔬",
        "title": "Discover Useful Image Features",
        "intro": (
            "Feature extraction converts visual information into useful "
            "descriptions. These features can help compare images, recognise "
            "patterns, or locate matching points between two images."
        ),
        "flow": ["Image", "Detect Features", "Describe Patterns", "Compare or Match"],
        "concepts": [
            ("Keypoint", "A distinctive location detected in an image."),
            ("ORB", "A feature method designed for efficient keypoint detection and binary descriptors."),
            ("Descriptor", "A numerical representation of a local image pattern."),
            ("Feature Matching", "Compares descriptors to find potentially corresponding features."),
            ("HOG", "Histogram of Oriented Gradients describes local gradient directions and magnitudes."),
        ],
        "fact": "Feature matching is useful for tasks such as image alignment, panorama creation, and visual recognition.",
        "question": "What is the purpose of a feature descriptor?",
        "options": [
            "To describe a local image pattern numerically",
            "To rename an image",
            "To change the monitor brightness",
            "To delete image pixels"
        ],
        "answer": "To describe a local image pattern numerically",
        "why": "Descriptors encode local visual patterns so they can be compared between images."
    },

    "Compression & Quality": {
        "icon": "📦",
        "title": "Smaller Files, Measurable Quality",
        "intro": (
            "Image compression reduces the number of bytes needed to store "
            "an image. Different formats and quality settings create different "
            "trade-offs between file size and visual similarity."
        ),
        "flow": ["Original Image", "Encode or Compress", "Smaller File", "Decode and Compare"],
        "concepts": [
            ("Lossless Compression", "Reduces file size without losing the original image information."),
            ("Lossy Compression", "Reduces file size by discarding some image information."),
            ("Compression Ratio", "Compares original file size with compressed file size."),
            ("PSNR", "A logarithmic measure based on pixel-level reconstruction error."),
            ("SSIM", "Estimates similarity using structural information such as luminance, contrast, and structure."),
        ],
        "fact": "A higher PSNR generally indicates lower pixel-level reconstruction error, but perceived image quality also depends on image content.",
        "question": "Which format commonly supports lossless image compression?",
        "options": ["PNG", "JPEG only", "MP3", "WAV"],
        "answer": "PNG",
        "why": "PNG uses lossless compression, so decoding preserves the stored pixel values."
    },
}


def render_learning_section(topic):
    """Render an interactive explanation and quiz for an existing topic."""

    content = LEARNING_CONTENT.get(topic)

    if not content:
        return

    st.markdown("---")
    st.markdown(
        f"""
        <div class="learning-heading">
            <div class="learning-kicker">PIXELLAB LEARNING STUDIO</div>
            <h2>{content['icon']} {content['title']}</h2>
            <p>Understand the science behind the result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("💡 Understand the Concept", expanded=False):
        st.markdown(f"**The idea in simple words**")
        st.write(content["intro"])

        st.markdown("#### The process, step by step")
        cols = st.columns(len(content["flow"]))

        for index, (col, step) in enumerate(zip(cols, content["flow"])):
            with col:
                st.markdown(
                    f"""
                    <div class="learning-step">
                        <span>STEP {index + 1}</span>
                        <div>{step}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("#### Key concepts")
        for name, description in content["concepts"]:
            with st.expander(name, expanded=False):
                st.write(description)

        st.info(f"🌟 **Did you know?** {content['fact']}")

    with st.expander("🧠 Test Your Understanding", expanded=False):
        st.write(content["question"])

        selected = st.radio(
            "Choose your answer:",
            content["options"],
            key=f"learning_quiz_{topic}",
            index=None,
        )

        if selected is not None:
            if selected == content["answer"]:
                st.success(f"Correct! {content['why']}")
            else:
                st.warning("Not quite! Try again, or review the concept above.")


def render_learning_hub():
    """Display the learning hub for all supported PixelLab topics."""

    st.markdown("## 🎓 PixelLab Learning Hub")
    st.write(
        "Explore the theory behind each image-processing technique. "
        "Choose a topic to review its concepts and test your understanding."
    )

    topics = list(LEARNING_CONTENT.keys())
    selected_topic = st.selectbox(
        "Choose a learning topic",
        topics,
        format_func=lambda topic: (
            f"{LEARNING_CONTENT[topic]['icon']} {topic}"
        ),
        key="learning_hub_topic",
    )

    render_learning_section(selected_topic)
