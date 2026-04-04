"""
PBR (Physically Based Rendering) material map generation from albedo textures.

Generates approximate normal, roughness, and metallic maps from a single
albedo/diffuse texture using image processing techniques. No ML models or
OpenCV dependency required -- only numpy and PIL.

These maps are consumed by game engines (Unity, Unreal, Blender) to produce
realistic PBR materials from a single input texture.
"""

import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D convolution on a single-channel image.

    Tries scipy.ndimage.convolve first for performance; falls back to a
    pure-numpy implementation when scipy is not available.
    """
    try:
        from scipy.ndimage import convolve
        return convolve(image, kernel, mode="reflect")
    except ImportError:
        pass

    # Pure-numpy fallback for 3x3 kernels (general case handled too).
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    h, w = image.shape
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[i : i + h, j : j + w]
    return output


# ---------------------------------------------------------------------------
# Normal map
# ---------------------------------------------------------------------------

def generate_normal_map(albedo: Image.Image, strength: float = 1.0) -> Image.Image:
    """Generate an approximate normal map from an albedo texture.

    The albedo is converted to grayscale as a height proxy.  Sobel filters
    compute the x/y gradients which are packed into the R and G channels of
    a tangent-space normal map (B channel = up = 1.0).

    Parameters
    ----------
    albedo : PIL.Image
        Input albedo / diffuse texture (any mode).
    strength : float
        Multiplier for the normal intensity.  Higher values exaggerate
        surface detail.

    Returns
    -------
    PIL.Image
        RGB normal map with values in [0, 255].
    """
    gray = np.array(albedo.convert("L"), dtype=np.float64) / 255.0

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    dx = _convolve2d(gray, sobel_x) * strength
    dy = _convolve2d(gray, sobel_y) * strength

    # Build tangent-space normal: (dx, dy, 1.0) then normalize.
    dz = np.ones_like(dx)
    length = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    length = np.maximum(length, 1e-8)

    nx = dx / length
    ny = dy / length
    nz = dz / length

    # Map from [-1, 1] to [0, 255].
    r = ((nx * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    g = ((ny * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    b = ((nz * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    normal = np.stack([r, g, b], axis=-1)
    logger.debug("Generated normal map (%dx%d, strength=%.2f)", albedo.width, albedo.height, strength)
    return Image.fromarray(normal, mode="RGB")


# ---------------------------------------------------------------------------
# Roughness map
# ---------------------------------------------------------------------------

def generate_roughness_map(albedo: Image.Image, base_roughness: float = 0.7) -> Image.Image:
    """Estimate a roughness map from the albedo texture.

    Local texture variance is used as a proxy for surface roughness:
    high-detail / high-contrast areas are treated as *shinier* (lower
    roughness) while smooth / low-contrast areas are *more matte* (higher
    roughness).

    Parameters
    ----------
    albedo : PIL.Image
        Input albedo / diffuse texture.
    base_roughness : float
        Base roughness value in [0, 1].  The final map is centred around
        this value.

    Returns
    -------
    PIL.Image
        Grayscale roughness map (bright = rough, dark = smooth/shiny).
    """
    gray = np.array(albedo.convert("L"), dtype=np.float64) / 255.0

    # Compute local variance with a uniform window (box filter).
    window_size = 5
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)

    local_mean = _convolve2d(gray, kernel)
    local_sq_mean = _convolve2d(gray ** 2, kernel)
    variance = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    # Normalize variance to [0, 1].
    var_max = variance.max()
    if var_max > 1e-8:
        variance_norm = variance / var_max
    else:
        variance_norm = variance

    # High variance -> shinier (lower roughness).
    roughness = base_roughness + (0.5 - variance_norm) * (1.0 - base_roughness) * 2
    roughness = np.clip(roughness, 0.0, 1.0)

    result = (roughness * 255).astype(np.uint8)
    logger.debug(
        "Generated roughness map (%dx%d, base_roughness=%.2f)",
        albedo.width, albedo.height, base_roughness,
    )
    return Image.fromarray(result, mode="L")


# ---------------------------------------------------------------------------
# Metallic map
# ---------------------------------------------------------------------------

def generate_metallic_map(albedo: Image.Image, threshold: float = 0.3) -> Image.Image:
    """Estimate a metallic map from the albedo texture.

    Metallic regions are approximated by looking for pixels with both high
    colour saturation *and* high brightness, which is a rough heuristic for
    metallic surfaces.  Most real-world objects are dielectric (non-metallic),
    so the output defaults to low values.

    Parameters
    ----------
    albedo : PIL.Image
        Input albedo / diffuse texture.
    threshold : float
        Combined saturation + brightness threshold below which pixels are
        considered non-metallic.  Lower values produce more metallic area.

    Returns
    -------
    PIL.Image
        Grayscale metallic map (bright = metallic, dark = dielectric).
    """
    rgb = np.array(albedo.convert("RGB"), dtype=np.float64) / 255.0

    # Per-pixel max and min across channels (manual HSV-like decomposition).
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    delta = cmax - cmin

    # Saturation: delta / max (0 where max == 0).
    saturation = np.where(cmax > 1e-8, delta / cmax, 0.0)

    # Brightness: simply the max channel value.
    brightness = cmax

    # Combined score -- both saturation and brightness must be high.
    score = saturation * brightness

    # Map score to metallic value, applying threshold.
    metallic = np.where(score > threshold, (score - threshold) / (1.0 - threshold + 1e-8), 0.0)
    metallic = np.clip(metallic, 0.0, 1.0)

    result = (metallic * 255).astype(np.uint8)
    logger.debug(
        "Generated metallic map (%dx%d, threshold=%.2f)",
        albedo.width, albedo.height, threshold,
    )
    return Image.fromarray(result, mode="L")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def generate_pbr_maps(
    albedo: Image.Image,
    strength: float = 1.0,
    base_roughness: float = 0.7,
) -> Dict[str, Image.Image]:
    """Generate all PBR maps from a single albedo texture.

    Parameters
    ----------
    albedo : PIL.Image
        Input albedo / diffuse texture.
    strength : float
        Normal map strength multiplier.
    base_roughness : float
        Base roughness value for the roughness map.

    Returns
    -------
    dict
        Mapping of ``"normal"``, ``"roughness"``, ``"metallic"`` to their
        respective PIL Images.
    """
    logger.info(
        "Generating PBR maps from albedo (%dx%d)",
        albedo.width, albedo.height,
    )

    pbr_maps: Dict[str, Image.Image] = {
        "normal": generate_normal_map(albedo, strength=strength),
        "roughness": generate_roughness_map(albedo, base_roughness=base_roughness),
        "metallic": generate_metallic_map(albedo),
    }

    logger.info(
        "PBR map generation complete: %s",
        ", ".join(pbr_maps.keys()),
    )
    return pbr_maps


def save_pbr_maps(pbr_maps: Dict[str, Image.Image], output_dir: str) -> Dict[str, str]:
    """Save PBR maps as PNG files.

    Parameters
    ----------
    pbr_maps : dict
        Mapping of map name to PIL Image (as returned by
        :func:`generate_pbr_maps`).
    output_dir : str
        Directory to write the PNG files into.  Created if it does not
        exist.

    Returns
    -------
    dict
        Mapping of map name to the absolute file path of the saved PNG.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}
    for name, img in pbr_maps.items():
        path = out / f"{name}_map.png"
        img.save(str(path))
        saved[name] = str(path)
        logger.info("Saved %s map to %s", name, path)

    return saved
