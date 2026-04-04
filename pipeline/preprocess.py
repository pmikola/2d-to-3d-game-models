"""
Image preprocessing for the 2D-to-3D pipeline.

Handles background removal, resizing, format normalization, and quality checks
to prepare images for Hi3DGen geometry generation.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Supported input formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# Quality thresholds
MIN_DIMENSION = 256
BLUR_THRESHOLD = 100.0  # Laplacian variance threshold


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from disk, handling various formats.

    Args:
        image_path: Path to the input image.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the image format is not supported.
        FileNotFoundError: If the image file does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    img = Image.open(image_path)
    logger.info(f"Loaded image: {image_path} ({img.size[0]}x{img.size[1]}, mode={img.mode})")
    return img


def convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert image to RGB, compositing RGBA onto white background.

    Args:
        img: Input PIL Image (any mode).

    Returns:
        PIL Image in RGB mode.
    """
    if img.mode == "RGBA":
        # Composite onto white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        logger.info("Converted RGBA to RGB (composited on white background).")
        return background
    elif img.mode == "RGB":
        return img
    else:
        logger.info(f"Converting image from {img.mode} to RGB.")
        return img.convert("RGB")


def check_image_quality(img: Image.Image) -> list[str]:
    """
    Check image quality and return warnings.

    Args:
        img: Input PIL Image.

    Returns:
        List of warning strings (empty if no issues).
    """
    warnings = []

    # Check dimensions
    w, h = img.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        warnings.append(
            f"Image is very small ({w}x{h}). Minimum recommended: {MIN_DIMENSION}x{MIN_DIMENSION}. "
            f"Results may be poor quality."
        )

    # Check for blur using Laplacian variance approximation
    # Convert to grayscale and apply edge detection
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges, dtype=np.float64)
    variance = edge_array.var()

    if variance < BLUR_THRESHOLD:
        warnings.append(
            f"Image appears blurry (edge variance: {variance:.1f}, threshold: {BLUR_THRESHOLD}). "
            f"Consider using a sharper image for better 3D reconstruction."
        )

    return warnings


def remove_background(img: Image.Image, use_gpu: bool = False) -> Image.Image:
    """
    Remove background from image using rembg.

    Hi3DGen works best with isolated objects on clean backgrounds.

    Args:
        img: Input PIL Image in RGB mode.
        use_gpu: Whether to use GPU acceleration for background removal.

    Returns:
        PIL Image with background removed (composited on white).
    """
    try:
        from rembg import remove

        logger.info("Removing background with rembg...")
        # rembg returns RGBA image
        result = remove(img)

        # Composite onto white background
        if result.mode == "RGBA":
            background = Image.new("RGB", result.size, (255, 255, 255))
            background.paste(result, mask=result.split()[3])
            result = background

        logger.info("Background removal complete.")
        return result

    except ImportError:
        logger.warning(
            "rembg not installed. Skipping background removal. "
            "Install with: pip install rembg[gpu] (GPU) or pip install rembg (CPU). "
            "Background removal significantly improves 3D reconstruction quality."
        )
        return img


def resize_and_pad(img: Image.Image, target_size: int = 512) -> Image.Image:
    """
    Resize image to target_size x target_size, maintaining aspect ratio with padding.

    Hi3DGen's NiRNE normal estimator uses 512x512 input internally.

    Args:
        img: Input PIL Image.
        target_size: Target dimension (default 512 to match Hi3DGen).

    Returns:
        PIL Image resized and padded to target_size x target_size.
    """
    w, h = img.size

    # Calculate scaling factor to fit within target_size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with high-quality resampling
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Pad to target_size x target_size (center the image on white background)
    padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    padded.paste(resized, (offset_x, offset_y))

    logger.info(f"Resized {w}x{h} -> {new_w}x{new_h}, padded to {target_size}x{target_size}.")
    return padded


def preprocess_image(
    image_path: str,
    target_size: int = 512,
    remove_bg: bool = True,
    use_gpu: bool = False,
) -> Image.Image:
    """
    Full preprocessing pipeline for a single image.

    Steps:
        1. Load image (handle PNG/JPG/WEBP/etc.)
        2. Convert to RGB if RGBA (composite on white)
        3. Quality check (warn on low-res or blurry)
        4. Remove background using rembg
        5. Resize to target_size x target_size (pad, don't stretch)

    Args:
        image_path: Path to the input image.
        target_size: Target dimension (default 512).
        remove_bg: Whether to remove background.
        use_gpu: Whether to use GPU for background removal.

    Returns:
        Preprocessed PIL Image ready for Hi3DGen.
    """
    # 1. Load
    img = load_image(image_path)

    # 2. Convert to RGB
    img = convert_to_rgb(img)

    # 3. Quality check
    warnings = check_image_quality(img)
    for warning in warnings:
        logger.warning(warning)

    # 4. Background removal
    if remove_bg:
        img = remove_background(img, use_gpu=use_gpu)

    # 5. Resize and pad
    img = resize_and_pad(img, target_size=target_size)

    return img
