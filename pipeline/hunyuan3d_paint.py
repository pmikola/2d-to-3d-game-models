"""Hunyuan3D-2.1 Paint pipeline for PBR texture generation with MMGP offloading.

Wraps the official Hunyuan3D Paint code to generate albedo, normal, roughness,
and metallic textures for a given geometry mesh.  MMGP (Multi-Modal GPU
Partitioning) offloading is used to keep peak VRAM within 14 GB on systems with
32 GB of RAM.

Reference: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
VRAM: ~14 GB peak with MMGP ``LowRAM_LowVRAM`` on a 32 GB RAM system.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Hunyuan3DPaintWrapper:
    """PBR texture generation using Hunyuan3D Paint + MMGP memory offloading."""

    def __init__(self, device_config, model_path=None, max_views=6, resolution=512, mmgp_profile="LowRAM_LowVRAM"):
        self.device_config = device_config
        self.model_path = model_path or "tencent/Hunyuan3D-2.1"
        self.max_views = max_views
        self.resolution = resolution
        self.mmgp_profile = mmgp_profile
        self.pipeline = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _find_paint_code(self) -> str | None:
        """Find ``hy3dpaint`` directory in common locations."""
        candidates = [
            Path("X:/Software Projects/2d-to-3d-game-models/Hunyuan3D-2.1"),
            Path.home() / ".cache" / "hunyuan3d" / "Hunyuan3D-2.1",
            Path("/content/Hunyuan3D-2.1"),
        ]
        for root in candidates:
            paint_dir = root / "hy3dpaint"
            if paint_dir.is_dir() and (paint_dir / "textureGenPipeline.py").exists():
                return str(root)
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self):
        """Load Hunyuan3D Paint with MMGP offloading."""
        if self._initialized:
            return

        import torch

        # Find and import paint pipeline code
        code_root = self._find_paint_code()
        if code_root is None:
            raise ImportError(
                "Hunyuan3D-2.1 paint code not found. Clone the repo:\n"
                "  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
            )

        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        from hy3dpaint.textureGenPipeline import (
            Hunyuan3DPaintConfig,
            Hunyuan3DPaintPipeline,
        )

        logger.info("Configuring Hunyuan3D Paint pipeline...")
        config = Hunyuan3DPaintConfig(max_num_view=self.max_views, resolution=self.resolution)
        config.device = "cuda" if self.device_config.has_gpu else "cpu"
        config.multiview_pretrained_path = self.model_path

        # Adjust paths relative to code root
        config.multiview_cfg_path = str(
            Path(code_root) / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"
        )
        config.custom_pipeline = str(
            Path(code_root) / "hy3dpaint" / "hunyuanpaintpbr"
        )
        config.realesrgan_ckpt_path = str(
            Path(code_root) / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
        )

        logger.info("Loading Hunyuan3D Paint pipeline...")
        self.pipeline = Hunyuan3DPaintPipeline(config)

        # Apply MMGP offloading to keep VRAM under budget
        try:
            from mmgp import offload, profile_type

            profile = getattr(profile_type, self.mmgp_profile, profile_type.LowRAM_LowVRAM)
            offload.profile(self.pipeline, profile)
            logger.info(f"MMGP offloading enabled ({self.mmgp_profile}).")
        except ImportError:
            logger.warning(
                "mmgp not installed. Paint pipeline runs without memory offloading. "
                "May OOM on GPUs < 24GB. Install: pip install mmgp"
            )

        self._initialized = True
        logger.info("Hunyuan3D Paint pipeline loaded.")

    def unload(self):
        """Remove paint pipeline from GPU."""
        from .device import unload_gpu_model

        unload_gpu_model("pipeline", self)
        self._initialized = False
        logger.info("Hunyuan3D Paint pipeline unloaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_textures(
        self,
        mesh_path,
        reference_image,
        output_dir,
        use_remesh: bool = True,
    ) -> dict:
        """Generate PBR textures for a mesh.

        Args:
            mesh_path: Path to GLB/OBJ mesh.
            reference_image: Original input PIL Image or path string.
            output_dir: Directory for textured output.
            use_remesh: Remesh before texturing.

        Returns:
            Dict with paths: textured_obj, albedo, normal, roughness, metallic.
        """
        import torch
        from .device import log_vram_status

        self.load()
        log_vram_status("paint_before_inference")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save reference image to file.
        # The paint pipeline loads the image from disk via Image.open() and
        # later passes it through RealESRGAN which internally calls
        # cv2.resize().  To avoid "src is not a numpy array" errors we must
        # ensure the saved file is a standard 8-bit RGB PNG that OpenCV can
        # handle without ambiguity.
        ref_path = str(output_path / "reference.png")
        if isinstance(reference_image, Image.Image):
            # Force to RGB uint8 so downstream cv2 calls receive clean data
            ref_img = reference_image.convert("RGB")
            ref_img.save(ref_path)
        elif isinstance(reference_image, np.ndarray):
            arr = reference_image
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(arr).convert("RGB").save(ref_path)
        else:
            # Assume it's already a path string
            ref_path = str(reference_image)
            if not Path(ref_path).exists():
                raise FileNotFoundError(
                    f"Reference image not found: {ref_path}"
                )

        obj_output = str(output_path / "textured.obj")

        logger.info(f"Running Hunyuan3D Paint (mesh={mesh_path})...")
        try:
            with torch.no_grad():
                result_path = self.pipeline(
                    mesh_path=str(mesh_path),
                    image_path=ref_path,
                    output_mesh_path=obj_output,
                    use_remesh=use_remesh,
                    save_glb=True,
                )
        except Exception as e:
            logger.error(f"Hunyuan3D Paint failed: {e}")
            raise

        log_vram_status("paint_after_inference")

        # Discover output files
        result = {
            "textured_obj": result_path or obj_output,
            "albedo": None,
            "normal": None,
            "roughness": None,
            "metallic": None,
        }

        # Scan for texture files produced by Paint
        for f in output_path.rglob("*"):
            fname = f.stem.lower()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                if "albedo" in fname or "diffuse" in fname or "basecolor" in fname:
                    result["albedo"] = str(f)
                elif "normal" in fname:
                    result["normal"] = str(f)
                elif "roughness" in fname:
                    result["roughness"] = str(f)
                elif "metallic" in fname or "metalness" in fname:
                    result["metallic"] = str(f)
                elif "mr" in fname or "metallicroughness" in fname:
                    # Combined metallic-roughness map
                    result["roughness"] = str(f)
                    result["metallic"] = str(f)

        # Fallback: if no individual albedo found, look for any texture image
        if result["albedo"] is None:
            for pattern in ["texture_atlas.*", "*.png"]:
                matches = list(output_path.glob(pattern))
                img_matches = [
                    m
                    for m in matches
                    if m.suffix.lower() in (".png", ".jpg")
                    and "normal" not in m.stem.lower()
                ]
                if img_matches:
                    result["albedo"] = str(img_matches[0])
                    break

        logger.info(f"Paint output: {result}")
        return result
