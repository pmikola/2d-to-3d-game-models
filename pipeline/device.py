"""
GPU/CPU detection, VRAM checking, and fallback logic.

Detects available hardware and configures the pipeline to use GPU (CUDA)
when available with sufficient VRAM, or falls back to CPU with adjusted parameters.
"""

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# Minimum VRAM thresholds in GB
VRAM_MIN_GEOMETRY = 8.0  # Hi3DGen needs ~8GB VRAM minimum
VRAM_MIN_TEXTURING = 10.0  # Text2Tex with SD2 needs ~10GB VRAM minimum
VRAM_COMFORTABLE = 16.0  # Comfortable for full pipeline


@dataclass
class DeviceConfig:
    """Hardware configuration for the pipeline."""

    device: torch.device
    dtype: torch.dtype
    has_gpu: bool
    vram_gb: float
    gpu_name: str
    # Adjusted parameters for resource constraints
    geometry_batch_size: int
    texture_num_viewpoints: int
    texture_ddim_steps: int
    use_xformers: bool

    def __str__(self) -> str:
        if self.has_gpu:
            return (
                f"GPU: {self.gpu_name} ({self.vram_gb:.1f} GB VRAM) | "
                f"dtype={self.dtype} | xformers={self.use_xformers}"
            )
        return f"CPU mode | dtype={self.dtype} | reduced parameters for performance"


def detect_device(force_cpu: bool = False) -> DeviceConfig:
    """
    Detect available hardware and return optimal configuration.

    Args:
        force_cpu: Force CPU mode even if GPU is available.

    Returns:
        DeviceConfig with hardware-appropriate settings.
    """
    if force_cpu or not torch.cuda.is_available():
        if not torch.cuda.is_available():
            logger.warning(
                "No CUDA GPU detected. Running on CPU — inference will be significantly slower."
            )
        else:
            logger.info("CPU mode forced by user.")

        return DeviceConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            has_gpu=False,
            vram_gb=0.0,
            gpu_name="N/A",
            geometry_batch_size=1,
            texture_num_viewpoints=8,  # Reduced from 36 for CPU
            texture_ddim_steps=25,  # Reduced from 50 for CPU
            use_xformers=False,
        )

    # GPU is available
    gpu_props = torch.cuda.get_device_properties(0)
    vram_gb = gpu_props.total_mem / (1024**3)
    gpu_name = gpu_props.name

    logger.info(f"Detected GPU: {gpu_name} with {vram_gb:.1f} GB VRAM")

    # Check for xformers availability
    use_xformers = False
    try:
        import xformers  # noqa: F401

        use_xformers = True
        logger.info("xformers detected — using memory-efficient attention.")
    except ImportError:
        logger.info("xformers not found — using standard attention.")

    if vram_gb < VRAM_MIN_GEOMETRY:
        logger.warning(
            f"GPU VRAM ({vram_gb:.1f} GB) is below minimum ({VRAM_MIN_GEOMETRY} GB). "
            f"Falling back to CPU for geometry generation."
        )
        return DeviceConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            has_gpu=False,
            vram_gb=vram_gb,
            gpu_name=gpu_name,
            geometry_batch_size=1,
            texture_num_viewpoints=8,
            texture_ddim_steps=25,
            use_xformers=False,
        )

    if vram_gb < VRAM_COMFORTABLE:
        logger.info(
            f"GPU VRAM ({vram_gb:.1f} GB) is moderate. Using reduced parameters."
        )
        return DeviceConfig(
            device=torch.device("cuda"),
            dtype=torch.float16,
            has_gpu=True,
            vram_gb=vram_gb,
            gpu_name=gpu_name,
            geometry_batch_size=1,
            texture_num_viewpoints=18,  # Reduced from 36
            texture_ddim_steps=35,  # Reduced from 50
            use_xformers=use_xformers,
        )

    # Comfortable VRAM — full parameters
    return DeviceConfig(
        device=torch.device("cuda"),
        dtype=torch.float16,
        has_gpu=True,
        vram_gb=vram_gb,
        gpu_name=gpu_name,
        geometry_batch_size=1,
        texture_num_viewpoints=36,
        texture_ddim_steps=50,
        use_xformers=use_xformers,
    )


def log_device_info(config: DeviceConfig) -> None:
    """Log detailed device information."""
    logger.info(f"Device configuration: {config}")
    if not config.has_gpu:
        logger.warning(
            "Running on CPU. Geometry generation may take 10-30 minutes per image. "
            "Texturing may take 5-15 minutes per viewpoint."
        )
    if config.has_gpu and config.vram_gb < VRAM_COMFORTABLE:
        logger.info(
            "Reduced viewpoints and DDIM steps to fit in available VRAM. "
            "Quality may be slightly lower than with 16+ GB VRAM."
        )
