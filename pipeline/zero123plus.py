"""Multi-view image generation using Zero123++ v1.2.

Generates 6 views from a single input image at predefined azimuths and
elevations.  The tiled 3x2 output is split into individual view images that
can be fed directly into Hunyuan3D-2mv for multi-view shape reconstruction.

Reference: https://github.com/SUDO-AI-3D/zero123plus
Model: https://huggingface.co/sudo-ai/zero123plus-v1.2
VRAM: ~6GB in fp16.
"""

import gc
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Zero123PlusWrapper:
    """Generate 6 multi-view images from a single input using Zero123++ v1.2.

    Views at azimuths: 30, 90, 150, 210, 270, 330 degrees.
    Elevations alternate: 20, -10 degrees.
    """

    # Zero123++ output azimuths (camera positions around the object).
    # Names use the **object-centric** convention used by Hunyuan3D-2mv's
    # MVImageProcessorV2: the label describes which side of the object the
    # camera *sees*, NOT where the camera is positioned.
    #
    #   Azimuth 90  -> camera at object's left  -> sees object's LEFT  -> "left"
    #   Azimuth 270 -> camera at object's right -> sees object's RIGHT -> "right"
    #
    # This matches MVImageProcessorV2.view2idx:
    #   {'front': 0, 'left': 1, 'back': 2, 'right': 3}
    AZIMUTH_MAP = {
        0: ("front_right_30", 30),
        1: ("left", 90),           # camera at 90 deg -> sees object's left
        2: ("back_right_150", 150),
        3: ("back", 210),
        4: ("right", 270),         # camera at 270 deg -> sees object's right
        5: ("front_left_330", 330),
    }

    # Cardinal views expected by Hunyuan3D-2mv MVImageProcessorV2
    CARDINAL_KEYS = {"front": 0, "left": 1, "back": 3, "right": 4}

    def __init__(self, device_config):
        self.device_config = device_config
        self.pipeline = None
        self._initialized = False

    def load(self):
        """Load Zero123++ v1.2 onto GPU."""
        if self._initialized:
            return

        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        logger.info("Loading Zero123++ v1.2...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing="trailing"
        )
        if self.device_config.has_gpu:
            self.pipeline.to("cuda")
        self._initialized = True
        logger.info("Zero123++ v1.2 loaded.")

    def unload(self):
        """Remove model from GPU."""
        from .device import unload_gpu_model

        unload_gpu_model("pipeline", self)
        self._initialized = False
        logger.info("Zero123++ unloaded.")

    @staticmethod
    def _prepare_for_zero123(
        image: Image.Image,
        target_size: int = 320,
        fill_ratio: float = 0.85,
    ) -> Image.Image:
        """Recenter and resize the foreground for Zero123++ v1.2.

        Zero123++ internally composites RGBA onto a mid-gray background using
        the alpha channel.  Passing an RGB image on white confuses the model
        because it cannot distinguish foreground from background, leading to
        shifted / cut-off multi-view outputs.

        This helper:
        1. Ensures the image has an alpha channel (uses rembg if needed).
        2. Finds the tight bounding box of the foreground (alpha > 0).
        3. Crops to that bounding box.
        4. Resizes to fill *fill_ratio* of a *target_size* x *target_size*
           canvas while preserving aspect ratio.
        5. Centers the result on a new RGBA canvas with transparent background.

        Args:
            image: Input PIL Image (RGB or RGBA).
            target_size: Output image dimension (default 320, the native
                resolution of Zero123++ v1.2).
            fill_ratio: How much of the canvas the object should fill
                (default 0.85).

        Returns:
            RGBA PIL Image of size *target_size* x *target_size*.
        """
        # --- 1. Ensure alpha channel ---
        if image.mode != "RGBA":
            try:
                from rembg import remove

                logger.info(
                    "Zero123++ prep: input has no alpha channel, running rembg..."
                )
                image = remove(image.convert("RGB"))
            except ImportError:
                logger.warning(
                    "rembg not installed; assuming white background for alpha "
                    "estimation.  Install rembg for better results."
                )
                # Heuristic: treat near-white pixels as background
                arr = np.array(image.convert("RGB"))
                white_thresh = 240
                bg_mask = np.all(arr >= white_thresh, axis=-1)
                alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
                image = image.convert("RGBA")
                image.putalpha(Image.fromarray(alpha))

        # --- 2. Find bounding box of foreground ---
        alpha_arr = np.array(image.split()[3])
        rows = np.any(alpha_arr > 0, axis=1)
        cols = np.any(alpha_arr > 0, axis=0)

        if not rows.any() or not cols.any():
            logger.warning(
                "Zero123++ prep: image is fully transparent, returning as-is."
            )
            return image.resize((target_size, target_size), Image.LANCZOS)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # --- 3. Crop to bounding box ---
        cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))

        # --- 4. Resize to fill_ratio of target_size ---
        cw, ch = cropped.size
        max_dim = int(target_size * fill_ratio)
        scale = max_dim / max(cw, ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        resized = cropped.resize((new_w, new_h), Image.LANCZOS)

        # --- 5. Center on transparent canvas ---
        canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        logger.info(
            f"Zero123++ prep: cropped ({cw}x{ch}) -> resized ({new_w}x{new_h}) "
            f"-> centered on {target_size}x{target_size} RGBA canvas."
        )
        return canvas

    def generate_views(
        self,
        image: Image.Image,
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
    ) -> dict:
        """Generate 6 views from a single input image.

        Returns:
            Dict with keys: front, left, back, right, front_right_30,
            back_right_150, front_left_330.  Values are PIL Images.
            Names use object-centric convention matching
            ``MVImageProcessorV2.view2idx``.
        """
        import torch
        from .device import log_vram_status

        self.load()
        log_vram_status("zero123_before_inference")

        # Prepare input: recenter, resize, keep RGBA for Zero123++.
        # Zero123++ internally composites RGBA onto mid-gray (127-128) via its
        # to_rgb_image() helper. Passing RGB-on-white defeats that logic and
        # produces shifted / cut-off views.
        image = self._prepare_for_zero123(image)

        logger.info(
            f"Running Zero123++ (steps={num_inference_steps}, "
            f"guidance={guidance_scale})..."
        )
        with torch.no_grad():
            result = self.pipeline(
                image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        tiled = result.images[0]  # 3x2 tiled image

        views_list = self._split_tiled_image(tiled)

        # Map to named dict
        views = {}
        for idx, (name, _az) in self.AZIMUTH_MAP.items():
            views[name] = views_list[idx]

        # Add cardinal alias: Zero123++ has no exact 0-deg front view;
        # the closest is the 30-deg "front_right_30" tile.
        # "left" (90 deg), "back" (210 deg), and "right" (270 deg) are
        # already present from the azimuth map with object-centric names
        # matching MVImageProcessorV2.view2idx.
        views["front"] = views["front_right_30"]

        log_vram_status("zero123_after_inference")
        logger.info(f"Generated {len(views)} views.")
        return views

    def _split_tiled_image(self, tiled: Image.Image) -> list:
        """Split 3x2 tiled output into 6 individual views."""
        w, h = tiled.size
        tile_w = w // 3
        tile_h = h // 2
        views = []
        for row in range(2):
            for col in range(3):
                box = (
                    col * tile_w,
                    row * tile_h,
                    (col + 1) * tile_w,
                    (row + 1) * tile_h,
                )
                views.append(tiled.crop(box))
        return views
