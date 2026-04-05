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

# Cached rembg session to avoid reloading the model on every call.
# Keys: model name (str), values: rembg session object.
_rembg_session_cache: dict = {}

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


def remove_background(
    img: Image.Image,
    use_gpu: bool = False,
    bg_model: str = "birefnet-general",
) -> Image.Image:
    """
    Remove background from image using rembg.

    Hi3DGen works best with isolated objects on clean backgrounds.
    By default uses the BiRefNet model (birefnet-general) which provides
    significantly better segmentation accuracy than the legacy U2Net default.
    Falls back to the default rembg model if the requested session cannot
    be created.

    Args:
        img: Input PIL Image in RGB mode.
        use_gpu: Whether to use GPU acceleration for background removal.
        bg_model: rembg session/model name (default ``"birefnet-general"``).
            Other useful values: ``"birefnet-general-lite"``,
            ``"birefnet-massive"``, ``"u2net"`` (legacy default).

    Returns:
        PIL Image in RGBA mode with background removed (alpha channel
        encodes the foreground mask).
    """
    try:
        from .deps import ensure_package

        ensure_package("rembg", pip_spec="rembg>=2.0.57")
        from rembg import new_session, remove

        # Reuse a cached session to avoid reloading the ONNX model on every
        # call.  BiRefNet model load can take 2-5 seconds, so caching the
        # session across images saves significant time in batch runs and
        # repeated single-image calls.
        session = _rembg_session_cache.get(bg_model)
        if session is None:
            try:
                session = new_session(bg_model)
                _rembg_session_cache[bg_model] = session
                logger.info(
                    "Created and cached rembg session (model=%s).", bg_model
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to create rembg session '{bg_model}': {exc}. "
                    "Falling back to default rembg model (u2net)."
                )
                session = None
        else:
            logger.info(
                "Reusing cached rembg session (model=%s).", bg_model
            )

        # rembg returns RGBA image
        if session is not None:
            result = remove(img, session=session)
        else:
            logger.info("Removing background with rembg (default model)...")
            result = remove(img)

        # Preserve the RGBA output directly — downstream consumers composite
        # onto their required background color at point of use.
        if result.mode != "RGBA":
            result = result.convert("RGBA")

        logger.info("Background removal complete (returning RGBA).")
        return result

    except ImportError:
        logger.warning(
            "rembg not installed. Skipping background removal. "
            "Install with: pip install rembg[gpu] (GPU) or pip install rembg (CPU). "
            "Background removal significantly improves 3D reconstruction quality."
        )
        # Return RGBA with full opacity so downstream code always gets RGBA
        if img.mode != "RGBA":
            return img.convert("RGBA")
        return img


def correct_dynamic_range(
    img: Image.Image,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> Image.Image:
    """Apply percentile-based dynamic range correction and adaptive gamma.

    The correction is computed on the Y (luminance) channel of YCbCr space so
    that colour hue and saturation are preserved.  When the image has an alpha
    channel, only foreground pixels (alpha > 0) are used for statistics, but the
    correction is applied to every pixel.

    Steps:
        1. Convert RGB to YCbCr via ``Y = 0.299*R + 0.587*G + 0.114*B``.
        2. Percentile-stretch Y to use the full 0-255 range.
        3. Apply adaptive gamma if mean Y is too dark (<70) or too bright (>200).
        4. Convert back to RGB, preserving Cb/Cr.

    Args:
        img: Input PIL Image (RGB or RGBA).
        low_percentile: Lower clipping percentile for stretch (default 1.0).
        high_percentile: Upper clipping percentile for stretch (default 99.0).

    Returns:
        Corrected PIL Image in the same mode as the input.
    """
    original_mode = img.mode
    alpha = None

    if img.mode == "RGBA":
        alpha = np.array(img.split()[3])
        rgb = np.array(img.convert("RGB"), dtype=np.float64)
        fg_mask = alpha > 0
    elif img.mode == "RGB":
        rgb = np.array(img, dtype=np.float64)
        fg_mask = None
    else:
        rgb = np.array(img.convert("RGB"), dtype=np.float64)
        fg_mask = None

    # --- RGB -> YCbCr (ITU-R BT.601) ---
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    # --- Percentile stretch on Y (foreground only for stats) ---
    if fg_mask is not None:
        y_fg = y[fg_mask]
    else:
        y_fg = y.ravel()

    if y_fg.size == 0:
        logger.debug("correct_dynamic_range: no foreground pixels, skipping.")
        return img

    p_low = np.percentile(y_fg, low_percentile)
    p_high = np.percentile(y_fg, high_percentile)

    if p_high - p_low < 1.0:
        logger.debug("correct_dynamic_range: near-constant luminance, skipping stretch.")
    else:
        y = np.clip((y - p_low) / (p_high - p_low) * 255.0, 0, 255)

    # --- Adaptive gamma ---
    if fg_mask is not None:
        mean_y = np.mean(y[fg_mask])
    else:
        mean_y = np.mean(y)

    if mean_y < 70 or mean_y > 200:
        # Adaptive gamma: map mean luminance to mid-gray (128).
        # Formula: gamma = log(0.5) / log(mean/255)
        normalized_mean = np.clip(mean_y / 255.0, 1e-6, 1.0 - 1e-6)
        gamma = np.log(0.5) / np.log(normalized_mean)
        # Clamp to a safe range to avoid extreme corrections
        gamma = float(np.clip(gamma, 0.3, 3.0))
        logger.info(f"Dynamic range: mean Y={mean_y:.1f}, applying adaptive gamma={gamma:.3f}")
        y = 255.0 * np.power(y / 255.0, gamma)
    else:
        logger.debug(f"Dynamic range: mean Y={mean_y:.1f}, gamma correction not needed.")

    # --- YCbCr -> RGB ---
    y_shifted = y - 0.0  # Y is already in 0-255
    cb_shifted = cb - 128.0
    cr_shifted = cr - 128.0

    r_out = y_shifted + 1.402 * cr_shifted
    g_out = y_shifted - 0.344136 * cb_shifted - 0.714136 * cr_shifted
    b_out = y_shifted + 1.772 * cb_shifted

    rgb_out = np.stack([r_out, g_out, b_out], axis=-1)
    rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)

    result = Image.fromarray(rgb_out, mode="RGB")

    # Restore alpha channel if the input had one
    if original_mode == "RGBA" and alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(Image.fromarray(alpha))

    return result


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

    # Pad to target_size x target_size.
    # Preserve RGBA if the input has an alpha channel (transparent padding);
    # otherwise use white padding for RGB images.
    if img.mode == "RGBA":
        padded = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        padded.paste(resized, (offset_x, offset_y))
    else:
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
    correct_exposure: bool = False,
    exposure_low_percentile: float = 1.0,
    exposure_high_percentile: float = 99.0,
    bg_model: str = "birefnet-general",
) -> Image.Image:
    """
    Full preprocessing pipeline for a single image.

    Steps:
        1. Load image (handle PNG/JPG/WEBP/etc.)
        2. Convert to RGB if RGBA (composite on white)
        3. Quality check (warn on low-res or blurry)
        4. Remove background using rembg
        4b. (Optional) Dynamic range / exposure correction
        5. Resize to target_size x target_size (pad, don't stretch)

    Args:
        image_path: Path to the input image.
        target_size: Target dimension (default 512).
        remove_bg: Whether to remove background.
        use_gpu: Whether to use GPU for background removal.
        correct_exposure: Apply dynamic range / exposure correction after
            background removal (default False).
        exposure_low_percentile: Low clipping percentile for dynamic range.
        exposure_high_percentile: High clipping percentile for dynamic range.
        bg_model: rembg session/model name for background removal
            (default ``"birefnet-general"``).  Pass ``"u2net"`` for the
            legacy model.

    Returns:
        Preprocessed PIL Image in RGBA mode (alpha from rembg preserved)
        ready for geometry generation.  Downstream consumers should
        composite onto their required background color at point of use
        via :func:`composite_on_background`.
    """
    # 1. Load (keep original mode — don't flatten alpha before bg removal)
    img = load_image(image_path)

    # 2. Quality check (works on any mode)
    warnings = check_image_quality(img)
    for warning in warnings:
        logger.warning(warning)

    # 3. Background removal — returns RGBA with alpha mask
    if remove_bg:
        # rembg expects RGB input; convert but don't flatten existing alpha yet
        img = remove_background(
            img.convert("RGB"), use_gpu=use_gpu, bg_model=bg_model
        )
    else:
        # No bg removal requested — ensure RGBA with full opacity
        if img.mode != "RGBA":
            img = img.convert("RGBA")

    # 4. Dynamic range correction (after bg removal, before resize)
    # Works on RGBA: uses alpha mask for foreground-only statistics
    if correct_exposure:
        logger.info("Applying dynamic range / exposure correction...")
        img = correct_dynamic_range(
            img,
            low_percentile=exposure_low_percentile,
            high_percentile=exposure_high_percentile,
        )

    # 5. Resize and pad (preserves RGBA with transparent padding)
    img = resize_and_pad(img, target_size=target_size)

    return img


def composite_on_background(
    img: Image.Image, bg_color: tuple = (255, 255, 255)
) -> Image.Image:
    """Composite an RGBA image onto a solid background color, returning RGB.

    Args:
        img: Input PIL Image (RGB or RGBA).
        bg_color: Background color as an (R, G, B) tuple (default white).

    Returns:
        PIL Image in RGB mode.
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    background = Image.new("RGB", img.size, bg_color)
    background.paste(img, mask=img.split()[3])
    return background


def remove_gray_background(
    img: Image.Image, gray_value: int = 128, tolerance: int = 30
) -> Image.Image:
    """Remove uniform gray background from MV-Adapter output, returning RGBA.

    MV-Adapter composites its views onto a mid-gray (128, 128, 128) canvas.
    This helper creates an alpha channel that marks those gray pixels as
    transparent so that Hunyuan3D-2mv's MVImageProcessorV2 receives proper
    RGBA input.

    Args:
        img: Input PIL Image (RGB expected from MV-Adapter output).
        gray_value: Central gray value to treat as background (default 128).
        tolerance: Per-channel tolerance around *gray_value* (default 30).

    Returns:
        PIL Image in RGBA mode with gray background made transparent.
    """
    arr = np.array(img.convert("RGB"))
    # Gray background: all channels within tolerance of gray_value
    is_bg = np.all(np.abs(arr.astype(int) - gray_value) < tolerance, axis=-1)
    alpha = np.where(is_bg, 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, mode="RGBA")
