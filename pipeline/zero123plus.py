"""Multi-view image generation using MV-Adapter (ICCV 2025).

Generates 6 views from a single input image at 768x768 resolution using
MV-Adapter on top of Stable Diffusion XL.  The output views can be fed
directly into Hunyuan3D-2mv for multi-view shape reconstruction.

Reference: https://github.com/huanngzh/MV-Adapter
Model weights: https://huggingface.co/huanngzh/mv-adapter
VRAM: ~14GB in fp16.
License: Apache 2.0.

Replaces the previous Zero123++ v1.2 backend (320x320, ~6GB).
"""

import gc
import logging
import math

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _to_float_tensor(values, device: str):
    """Convert a list/array/tensor to float32 on the target device."""
    import torch

    if isinstance(values, (list, tuple, np.ndarray)):
        return torch.tensor(values, dtype=torch.float32, device=device)
    return values.to(device=device, dtype=torch.float32)


def _build_orthographic_camera_c2w(
    elevation_deg,
    distance,
    azimuth_deg,
    device: str,
):
    """
    Recreate MV-Adapter's orthographic camera poses without importing its full
    mesh utilities package, which also pulls in optional rasterizer deps.
    """
    import torch
    import torch.nn.functional as F

    azimuth_deg = _to_float_tensor(azimuth_deg, device=device)
    elevation_deg = _to_float_tensor(elevation_deg, device=device)
    distance = _to_float_tensor(distance, device=device)

    elevation = elevation_deg * math.pi / 180.0
    azimuth = azimuth_deg * math.pi / 180.0
    camera_positions = torch.stack(
        [
            distance * torch.cos(elevation) * torch.cos(azimuth),
            distance * torch.cos(elevation) * torch.sin(azimuth),
            distance * torch.sin(elevation),
        ],
        dim=-1,
    )

    num_views = camera_positions.shape[0]
    center = torch.zeros_like(camera_positions)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).repeat(
        num_views, 1
    )
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)

    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w


def _get_opencv_from_blender(matrix_world):
    """Match MV-Adapter's Blender-to-OpenCV camera conversion."""
    import torch

    opencv_world_to_cam = torch.linalg.inv(matrix_world)
    opencv_world_to_cam[1, :] *= -1
    opencv_world_to_cam[2, :] *= -1
    return opencv_world_to_cam[:3, :3], opencv_world_to_cam[:3, 3]


def _get_plucker_embeds_from_cameras_ortho(c2w, image_size: int):
    """
    Recreate MV-Adapter's orthographic Plucker embedding helper locally so the
    full `mvadapter.utils` package is not imported during inference setup.
    """
    import torch
    import torch.nn.functional as F

    plucker_embeds = []
    for cam_matrix in c2w:
        R, T = _get_opencv_from_blender(cam_matrix)
        cam_pos = -R.T @ T
        view_dir = R.T @ torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=cam_matrix.device
        )
        cam_pos = F.normalize(cam_pos, dim=0)
        plucker = torch.cat([view_dir, cam_pos])
        plucker = plucker.unsqueeze(-1).unsqueeze(-1).repeat(1, image_size, image_size)
        plucker_embeds.append(plucker)

    return torch.stack(plucker_embeds)


class MVAdapterWrapper:
    """Generate selected multi-view images from a single input using MV-Adapter.

    Views at azimuths: 0, 45, 90, 180, 270, 315 degrees (6 views at 768x768).

    MV-Adapter generates higher resolution (768x768 vs 320x320) and more
    view-consistent multi-view images than Zero123++, at the cost of higher
    VRAM usage (~14GB vs ~6GB).
    """

    # MV-Adapter default azimuths and their cardinal names.
    # Uses the **object-centric** convention used by Hunyuan3D-2mv's
    # MVImageProcessorV2: the label describes which side of the object the
    # camera *sees*, NOT where the camera is positioned.
    #
    # Per the MV-Adapter paper (arXiv 2412.03632, Figures 12-14):
    #
    #   "The azimuth angles of the images from left to right are
    #    0, 45, 90, 180, 270, 315, corresponding to the front,
    #    front-left, left, back, right, and front-right of the object."
    #
    # The -90 degree offset applied in ``_setup_cameras`` is part of the
    # camera *construction* math (matching MV-Adapter's own inference
    # script). The camera position is computed as:
    #   (cos(az - 90°), sin(az - 90°), 0)
    #
    # Empirically confirmed view mapping (effective world azimuths after offset):
    #   Input az   0° -> effective -90° -> camera at (0, -1) -> FRONT  (confirmed)
    #   Input az  45° -> effective -45° -> camera at (+0.7, -0.7) -> FRONT-RIGHT diagonal
    #   Input az  90° -> effective   0° -> camera at (+1, 0)      -> RIGHT side (confirmed wrong as "left")
    #   Input az 180° -> effective +90° -> camera at (0, +1)      -> BACK  (confirmed)
    #   Input az 270° -> effective +180°-> camera at (-1, 0)      -> LEFT side (confirmed wrong as "right")
    #   Input az 315° -> effective +225°-> camera at (-0.7, -0.7) -> FRONT-LEFT diagonal
    #
    # The paper's label convention has left/right swapped relative to the
    # empirical output.  The AZIMUTH_MAP below uses the empirically correct
    # labels so CARDINAL_KEYS selects the right output images.
    #
    # MVImageProcessorV2.view2idx = {'front': 0, 'left': 1, 'back': 2, 'right': 3}
    AZIMUTH_DEG = [0, 45, 90, 180, 270, 315]
    AZIMUTH_MAP = {
        0: ("front", 0),            # confirmed: sees object's front
        1: ("front_right", 45),     # diagonal (front-right in world space)
        2: ("right", 90),           # confirmed empirically: sees object's RIGHT (paper says "left")
        3: ("back", 180),           # confirmed: sees object's back
        4: ("left", 270),           # confirmed empirically: sees object's LEFT  (paper says "right")
        5: ("front_left", 315),     # diagonal (front-left in world space)
    }

    # Cardinal views expected by Hunyuan3D-2mv MVImageProcessorV2
    CARDINAL_KEYS = {"front": 0, "left": 4, "back": 3, "right": 2}

    # Output resolution
    OUTPUT_SIZE = 768

    # Models
    BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
    ADAPTER_REPO = "huanngzh/mv-adapter"
    ADAPTER_WEIGHT = "mvadapter_i2mv_sdxl.safetensors"

    VIEW_NAME_TO_INDEX = {name: idx for idx, (name, _az) in AZIMUTH_MAP.items()}

    def __init__(self, device_config):
        self.device_config = device_config
        self.pipeline = None
        self._initialized = False
        self._loaded_num_views = None

    def load(self, num_views: int | None = None):
        """Load SDXL + MV-Adapter onto GPU."""
        target_num_views = num_views or len(self.AZIMUTH_DEG)

        if self._initialized and self._loaded_num_views == target_num_views:
            return
        if self._initialized and self._loaded_num_views != target_num_views:
            self.unload()

        # Auto-install MV-Adapter if missing (not on PyPI — cloned from GitHub).
        from .deps import ensure_package

        ensure_package(
            "mvadapter",
            git_url="https://github.com/huanngzh/MV-Adapter",
            allow_no_deps_fallback=True,
        )

        import torch
        from diffusers import AutoencoderKL, DDPMScheduler

        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import (
            MVAdapterI2MVSDXLPipeline,
        )
        from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

        logger.info("Loading MV-Adapter (SDXL base + i2mv adapter)...")

        # Load SDXL with fp16 VAE fix
        vae = AutoencoderKL.from_pretrained(self.VAE_MODEL)
        self.pipeline = MVAdapterI2MVSDXLPipeline.from_pretrained(
            self.BASE_MODEL,
            vae=vae,
            torch_dtype=torch.float16,
        )

        # Configure shifted SNR scheduler for better multi-view quality
        self.pipeline.scheduler = ShiftSNRScheduler.from_scheduler(
            self.pipeline.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=DDPMScheduler,
        )

        # Load MV-Adapter weights
        num_views = target_num_views
        self.pipeline.init_custom_adapter(num_views=num_views)
        self.pipeline.load_custom_adapter(
            self.ADAPTER_REPO, weight_name=self.ADAPTER_WEIGHT
        )

        # Move to device
        device = "cuda" if self.device_config.has_gpu else "cpu"
        dtype = torch.float16 if self.device_config.has_gpu else torch.float32
        self.pipeline.to(dtype=dtype)

        # NOTE:
        # MV-Adapter installs custom UNet attention processors that cache
        # reference hidden states by processor name. Diffusers'
        # enable_model_cpu_offload() changes the execution/hook path enough that
        # the reference-cache pass becomes incomplete on this pipeline, causing:
        #   KeyError: '...attn1.processor'
        # So we keep MV-Adapter on the GPU directly and control memory with
        # fp16, VAE slicing, and reduced step counts instead of CPU offload.
        self.pipeline.to(device=device, dtype=dtype)

        self.pipeline.cond_encoder.to(device=device, dtype=dtype)

        # MV-Adapter passes custom cross-attention kwargs through the UNet.
        # Diffusers attention slicing swaps in an attention processor that
        # ignores those kwargs, so we keep it disabled here.
        try:
            self.pipeline.vae.enable_slicing()
        except Exception:
            # Older diffusers versions still expose enable_vae_slicing().
            self.pipeline.enable_vae_slicing()

        self._initialized = True
        self._loaded_num_views = target_num_views
        logger.info("MV-Adapter loaded successfully.")

    def unload(self):
        """Remove model from GPU and free VRAM."""
        from .device import unload_gpu_model

        unload_gpu_model("pipeline", self)
        self._initialized = False
        self._loaded_num_views = None
        logger.info("MV-Adapter unloaded.")

    def _resolve_view_indices(
        self,
        requested_view_names: list[str] | None = None,
    ) -> list[int]:
        """Resolve requested view names to ordered MV-Adapter indices."""
        if requested_view_names is None:
            return list(self.AZIMUTH_MAP.keys())

        indices = []
        for name in requested_view_names:
            idx = self.VIEW_NAME_TO_INDEX.get(name)
            if idx is None:
                valid_names = ", ".join(sorted(self.VIEW_NAME_TO_INDEX))
                raise ValueError(
                    f"Unknown MV-Adapter view '{name}'. Valid names: {valid_names}"
                )
            indices.append(idx)
        return indices

    @staticmethod
    def _prepare_for_mvadapter(
        image: Image.Image,
        target_size: int = 768,
        fill_ratio: float = 0.9,
    ) -> Image.Image:
        """Recenter and resize the foreground for MV-Adapter.

        MV-Adapter expects an RGB image on a neutral gray (0.5) background,
        with the foreground object centered and filling most of the canvas.

        The preprocessing pipeline now provides RGBA images with a proper
        alpha channel from rembg, so this helper uses that alpha directly
        (no redundant rembg re-run or white-threshold heuristic).

        Steps:
        1. Use the provided alpha channel (convert to RGBA with full
           opacity if the input is somehow RGB).
        2. Find the tight bounding box of the foreground (alpha > 0).
        3. Crop to that bounding box.
        4. Resize to fill *fill_ratio* of a *target_size* x *target_size*
           canvas while preserving aspect ratio.
        5. Composite the result onto a mid-gray (128, 128, 128) RGB canvas,
           matching MV-Adapter's expected preprocessing.

        Args:
            image: Input PIL Image (RGBA expected from preprocessing;
                RGB is handled as a fallback with full opacity).
            target_size: Output image dimension (default 768, native
                resolution of MV-Adapter on SDXL).
            fill_ratio: How much of the canvas the object should fill
                (default 0.9, matching MV-Adapter's preprocess_image).

        Returns:
            RGB PIL Image of size *target_size* x *target_size* on gray bg.
        """
        # --- 1. Use the provided alpha channel ---
        if image.mode != "RGBA":
            logger.warning(
                "MV-Adapter prep: expected RGBA input but got %s; "
                "converting with full opacity.",
                image.mode,
            )
            image = image.convert("RGBA")

        # --- 2. Find bounding box of foreground ---
        alpha_arr = np.array(image.split()[3])
        rows = np.any(alpha_arr > 0, axis=1)
        cols = np.any(alpha_arr > 0, axis=0)

        if not rows.any() or not cols.any():
            logger.warning(
                "MV-Adapter prep: image is fully transparent, returning gray canvas."
            )
            return Image.new("RGB", (target_size, target_size), (128, 128, 128))

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # --- 3. Crop to bounding box (with 1px margin) ---
        y_min = max(y_min - 1, 0)
        x_min = max(x_min - 1, 0)
        y_max = min(y_max + 1, alpha_arr.shape[0])
        x_max = min(x_max + 1, alpha_arr.shape[1])
        cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))

        # --- 4. Resize to fill_ratio of target_size ---
        cw, ch = cropped.size
        max_dim = int(target_size * fill_ratio)
        if ch > cw:
            new_w = int(cw * max_dim / ch)
            new_h = max_dim
        else:
            new_h = int(ch * max_dim / cw)
            new_w = max_dim
        resized = cropped.resize((new_w, new_h), Image.LANCZOS)

        # --- 5. Composite onto gray background ---
        # MV-Adapter expects: RGB = foreground * alpha + 0.5 * (1 - alpha)
        # This matches the preprocess_image function in the official repo.
        arr = np.array(resized).astype(np.float32) / 255.0

        # Place on full canvas
        canvas = np.zeros((target_size, target_size, 4), dtype=np.float32)
        start_h = (target_size - new_h) // 2
        start_w = (target_size - new_w) // 2
        canvas[start_h : start_h + new_h, start_w : start_w + new_w] = arr

        # Composite: RGB * A + gray * (1 - A)
        rgb = canvas[:, :, :3]
        alpha = canvas[:, :, 3:4]
        composited = rgb * alpha + 0.5 * (1.0 - alpha)
        composited = (composited * 255).clip(0, 255).astype(np.uint8)

        result = Image.fromarray(composited, "RGB")

        logger.info(
            f"MV-Adapter prep: cropped ({cw}x{ch}) -> resized ({new_w}x{new_h}) "
            f"-> centered on {target_size}x{target_size} gray-bg RGB canvas."
        )
        return result

    def _setup_cameras(
        self,
        indices: list[int] | None = None,
        device: str = "cuda",
    ):
        """Create camera embeddings for the requested view indices.

        Returns the Plucker embedding control images expected by MV-Adapter.
        Camera azimuths follow the official MV-Adapter convention with a -90
        degree offset applied internally by get_orthogonal_camera.
        """
        import torch

        selected_indices = indices or list(self.AZIMUTH_MAP.keys())
        azimuths = [self.AZIMUTH_MAP[idx][1] for idx in selected_indices]
        num_views = len(azimuths)

        c2w = _build_orthographic_camera_c2w(
            elevation_deg=[0] * num_views,
            distance=[1.8] * num_views,
            azimuth_deg=[x - 90 for x in azimuths],
            device=device,
        )

        plucker_embeds = _get_plucker_embeds_from_cameras_ortho(c2w, self.OUTPUT_SIZE)
        control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

        return control_images

    def generate_views(
        self,
        image: Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        requested_view_names: list[str] | None = None,
        batch_size: int | None = None,
    ) -> dict:
        """Generate selected views from a single input image.

        Args:
            image: Input PIL Image (RGB or RGBA).
            num_inference_steps: Number of diffusion steps (default 50).
            guidance_scale: Classifier-free guidance scale (default 3.0).
            requested_view_names: Optional subset of named views to generate.
            batch_size: Optional sequential sub-batch size. Smaller values lower
                peak activation memory but add some extra runtime.

        Returns:
            Dict keyed by the requested object-centric view names.
        """
        import torch

        from .device import log_vram_status, release_runtime_memory

        selected_indices = self._resolve_view_indices(requested_view_names)
        total_views = len(selected_indices)
        if batch_size is None or batch_size <= 0:
            batch_size = total_views

        self.load(num_views=min(batch_size, total_views))
        log_vram_status("mvadapter_before_inference")

        # Prepare input: recenter, resize, composite on gray background.
        reference_image = self._prepare_for_mvadapter(
            image, target_size=self.OUTPUT_SIZE
        )

        device = "cuda" if self.device_config.has_gpu else "cpu"
        views = {}
        for chunk_id, chunk_start in enumerate(range(0, total_views, batch_size), start=1):
            chunk_indices = selected_indices[chunk_start : chunk_start + batch_size]
            if self._loaded_num_views != len(chunk_indices):
                self.load(num_views=len(chunk_indices))

            control_images = self._setup_cameras(indices=chunk_indices, device=device)
            chunk_names = [self.AZIMUTH_MAP[idx][0] for idx in chunk_indices]

            logger.info(
                "Running MV-Adapter chunk %d (steps=%d, guidance=%.1f, views=%s, "
                "resolution=%dx%d)...",
                chunk_id,
                num_inference_steps,
                guidance_scale,
                chunk_names,
                self.OUTPUT_SIZE,
                self.OUTPUT_SIZE,
            )

            with torch.no_grad():
                result = self.pipeline(
                    "high quality",
                    height=self.OUTPUT_SIZE,
                    width=self.OUTPUT_SIZE,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=len(chunk_indices),
                    control_image=control_images,
                    control_conditioning_scale=1.0,
                    reference_image=reference_image,
                    reference_conditioning_scale=1.0,
                    negative_prompt=(
                        "watermark, ugly, deformed, noisy, blurry, low contrast"
                    ),
                )

            for idx, image_out in zip(chunk_indices, result.images):
                name, _az = self.AZIMUTH_MAP[idx]
                views[name] = image_out

            del result
            del control_images
            release_runtime_memory(f"after_mvadapter_chunk_{chunk_id}")

        log_vram_status("mvadapter_after_inference")
        logger.info(f"Generated {len(views)} views at {self.OUTPUT_SIZE}x{self.OUTPUT_SIZE}.")
        return views


# Backward-compatible alias so old imports still work.
Zero123PlusWrapper = MVAdapterWrapper
