"""Multi-view image generation using CharacterGen's 2D stage.

Generates 4 orthogonal views from a single input image using the
CharacterGen 2D stage pipeline (Tune-A-Video architecture with multi-view
conditioning).

Reference: https://github.com/zjp-shadow/CharacterGen
Model weights: https://huggingface.co/zjpshadow/CharacterGen
VRAM: ~8-10GB in fp16 (estimated, 4-view generation at 512x768).
License: See CharacterGen repository.

This wrapper provides the same interface as MVAdapterWrapper so it can be
used as a drop-in replacement for multi-view generation in the full
pipeline backend.

Prerequisites
-------------
CharacterGen is NOT a simple pip-installable package.  It requires:
  1. A clone of https://github.com/zjp-shadow/CharacterGen
  2. Model weights from HuggingFace (zjpshadow/CharacterGen)
  3. The ``skytnt/anime-seg`` ONNX model for background removal
  4. Pose reference images and camera matrices from the repo's
     ``2D_Stage/material/`` directory.

The ``load()`` method checks for these prerequisites and raises a clear
error message if anything is missing.  Auto-installation is limited to
cloning the repo and downloading weights via ``huggingface-cli``; complex
system dependencies (if any) are left to the user.
"""

import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default location for the CharacterGen clone, relative to the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CHARGEN_PATH = _PROJECT_ROOT / "third_party" / "CharacterGen"

# HuggingFace repo that hosts the 2D stage weights.
CHARGEN_HF_REPO = "zjpshadow/CharacterGen"

# Pretrained SD2.1 base used by CharacterGen's UNet.
CHARGEN_BASE_MODEL = "stabilityai/stable-diffusion-2-1"

# CLIP image encoder used for reference conditioning.
CHARGEN_IMAGE_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class CharacterGenMVWrapper:
    """Generate 4 orthogonal views from a single image using CharacterGen.

    Produces front / left / back / right views compatible with Hunyuan3D-2mv.

    The interface mirrors ``MVAdapterWrapper`` so the orchestrator can swap
    between them transparently:

        wrapper = CharacterGenMVWrapper(device_config, chargen_path=...)
        wrapper.load()
        views = wrapper.generate_views(image)  # dict[str, PIL.Image]
        wrapper.unload()
    """

    # CharacterGen generates 4 views ordered as described in its pose.json.
    # Based on the camera matrices in material/pose.json, the 4 views
    # correspond to: front (0deg), right (90deg), back (180deg), left (270deg).
    # This mapping aligns them to Hunyuan3D-2mv's expected naming.
    VIEW_ORDER = ["front", "right", "back", "left"]

    # Output resolution: CharacterGen produces 512x768 images (WxH).
    OUTPUT_WIDTH = 512
    OUTPUT_HEIGHT = 768

    def __init__(
        self,
        device_config,
        chargen_path: str | None = None,
    ):
        """
        Args:
            device_config: DeviceConfig from pipeline.device.
            chargen_path: Path to a cloned CharacterGen repository.
                If None, defaults to ``third_party/CharacterGen`` under the
                project root.
        """
        self.device_config = device_config
        self.chargen_path = Path(chargen_path) if chargen_path else _DEFAULT_CHARGEN_PATH
        self._stage_dir = self.chargen_path / "2D_Stage"

        # Lazy-loaded model components (set in load()).
        self._pipeline = None
        self._vae = None
        self._text_encoder = None
        self._image_encoder = None
        self._feature_extractor = None
        self._unet = None
        self._ref_unet = None
        self._tokenizer = None
        self._pose_guider = None
        self._camera_matrixs = None
        self._pose_imgs = None
        self._initialized = False

        # Config values from CharacterGen's infer.yaml
        self._unet_condition_type = "image"
        self._use_noise = False
        self._use_shifted_noise = False
        self._noise_d = 0.0
        self._video_length = 4
        self._guidance_scale = 5.0

    # ------------------------------------------------------------------
    # Prerequisite checks
    # ------------------------------------------------------------------

    def _check_prerequisites(self) -> None:
        """Verify that all CharacterGen prerequisites are in place.

        Raises a descriptive RuntimeError if anything is missing so the
        user knows exactly what to fix.
        """
        errors = []

        if not self.chargen_path.is_dir():
            errors.append(
                f"CharacterGen repository not found at: {self.chargen_path}\n"
                f"  Clone it with:\n"
                f"    git clone https://github.com/zjp-shadow/CharacterGen "
                f"{self.chargen_path}"
            )

        stage_dir = self._stage_dir
        if self.chargen_path.is_dir() and not stage_dir.is_dir():
            errors.append(
                f"2D_Stage directory not found at: {stage_dir}\n"
                f"  The repository may be incomplete. Try re-cloning."
            )

        # Check for model weights.
        ckpt_dir = stage_dir / "models" / "checkpoint"
        if stage_dir.is_dir() and not ckpt_dir.is_dir():
            errors.append(
                f"Model weights not found at: {ckpt_dir}\n"
                f"  Download with:\n"
                f"    huggingface-cli download --resume-download "
                f"{CHARGEN_HF_REPO} --include '2D_Stage/*' "
                f"--local-dir {self.chargen_path}"
            )

        # Check for pose material.
        material_dir = stage_dir / "material"
        pose_json = material_dir / "pose.json"
        if stage_dir.is_dir() and not pose_json.is_file():
            errors.append(
                f"Pose material not found at: {pose_json}\n"
                f"  Ensure the full CharacterGen repository is cloned "
                f"(including 2D_Stage/material/)."
            )

        if errors:
            sep = "\n\n"
            raise RuntimeError(
                "CharacterGen prerequisites not met:\n\n"
                + sep.join(errors)
                + "\n\nSee https://github.com/zjp-shadow/CharacterGen for "
                "full setup instructions."
            )

    # ------------------------------------------------------------------
    # Model loading / unloading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load CharacterGen 2D stage models onto GPU.

        This performs the following steps:
          1. Check prerequisites (repo clone, weights, pose data).
          2. Add 2D_Stage to sys.path so its modules can be imported.
          3. Load CLIP tokenizer/encoder, VAE, UNet, RefUNet, PoseGuider.
          4. Load checkpoint weights from HuggingFace-downloaded files.
          5. Prepare camera matrices and pose conditioning images.
          6. Build the TuneAVideoPipeline for inference.
        """
        if self._initialized:
            return

        self._check_prerequisites()

        import sys

        import torch

        # Add 2D_Stage to sys.path so we can import tuneavideo modules.
        stage_dir_str = str(self._stage_dir)
        if stage_dir_str not in sys.path:
            sys.path.insert(0, stage_dir_str)

        logger.info("Loading CharacterGen 2D stage models...")

        # ---------- determine device and dtype ----------
        device = "cuda" if self.device_config.has_gpu else "cpu"
        # CharacterGen uses bfloat16 on compute capability >= 8,
        # float16 on older GPUs, float32 on CPU.
        if device == "cpu":
            weight_dtype = torch.float32
        elif torch.cuda.get_device_capability()[0] >= 8:
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float16

        # ---------- paths ----------
        ckpt_dir = self._stage_dir / "models" / "checkpoint"
        image_encoder_path = self._stage_dir / "models" / "image_encoder"
        pretrained_model_path = CHARGEN_BASE_MODEL
        material_dir = self._stage_dir / "material"

        # If the image encoder is not locally available, fall back to HF.
        if not image_encoder_path.is_dir():
            image_encoder_path = CHARGEN_IMAGE_ENCODER

        # ---------- load config ----------
        config_path = self._stage_dir / "configs" / "infer.yaml"
        if config_path.is_file():
            import yaml

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            validation = cfg.get("validation", {})
            self._guidance_scale = validation.get("guidance_scale", 5.0)
            self._video_length = validation.get("video_length", 4)
            self._use_noise = cfg.get("use_noise", False)
            self._use_shifted_noise = cfg.get("use_shifted_noise", False)
            self._unet_condition_type = cfg.get("unet_condition_type", "image")

            # Override paths from config if they are relative to 2D_Stage.
            cfg_ckpt = cfg.get("ckpt_dir")
            if cfg_ckpt:
                resolved = (self._stage_dir / cfg_ckpt).resolve()
                if resolved.is_dir():
                    ckpt_dir = resolved
            cfg_ie = cfg.get("image_encoder_path")
            if cfg_ie:
                resolved = (self._stage_dir / cfg_ie).resolve()
                if resolved.is_dir():
                    image_encoder_path = resolved

        # ---------- load CLIP / VAE / UNet ----------
        from transformers import (
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTokenizer,
            CLIPVisionModelWithProjection,
        )

        from diffusers import AutoencoderKL

        # Import CharacterGen-specific UNet variants from the cloned repo.
        from tuneavideo.models.unet_mv2d_condition import UNetMV2DConditionModel
        from tuneavideo.models.unet_mv2d_ref import UNetMV2DRefModel

        logger.info("  Loading CLIP tokenizer and text encoder...")
        self._tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        self._text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )

        logger.info("  Loading CLIP image encoder...")
        self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(image_encoder_path)
        )
        self._feature_extractor = CLIPImageProcessor()

        logger.info("  Loading VAE...")
        self._vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae"
        )

        # Determine UNet kwargs from config.
        unet_kwargs = {}
        if config_path.is_file():
            import yaml

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            unet_additional = cfg.get("unet_additional_kwargs", {})
            if unet_additional:
                unet_kwargs = unet_additional

        logger.info("  Loading UNet (MV2D condition model)...")
        self._unet = UNetMV2DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            **unet_kwargs,
        )

        logger.info("  Loading RefUNet...")
        self._ref_unet = UNetMV2DRefModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            **unet_kwargs,
        )

        # ---------- load checkpoint weights ----------
        logger.info("  Loading checkpoint weights from %s...", ckpt_dir)
        unet_ckpt = ckpt_dir / "pytorch_model.bin"
        ref_unet_ckpt = ckpt_dir / "pytorch_model_1.bin"
        pose_guider_ckpt = ckpt_dir / "pytorch_model_2.bin"

        if unet_ckpt.is_file():
            params = torch.load(str(unet_ckpt), map_location="cpu")
            self._unet.load_state_dict(params)
            del params
        else:
            logger.warning("UNet checkpoint not found: %s", unet_ckpt)

        if ref_unet_ckpt.is_file():
            params = torch.load(str(ref_unet_ckpt), map_location="cpu")
            self._ref_unet.load_state_dict(params)
            del params
        else:
            logger.warning("RefUNet checkpoint not found: %s", ref_unet_ckpt)

        # ---------- optional PoseGuider ----------
        if pose_guider_ckpt.is_file():
            try:
                from tuneavideo.models.PoseGuider import PoseGuider

                self._pose_guider = PoseGuider(noise_latent_channels=320)
                pg_params = torch.load(str(pose_guider_ckpt), map_location="cpu")
                self._pose_guider.load_state_dict(pg_params)
                del pg_params
                self._pose_guider.to(device=device, dtype=weight_dtype)
                logger.info("  PoseGuider loaded.")
            except Exception as exc:
                logger.warning("PoseGuider load failed (%s); proceeding without it.", exc)
                self._pose_guider = None
        else:
            logger.info("  No PoseGuider checkpoint found; skipping.")
            self._pose_guider = None

        # ---------- move models to device ----------
        self._text_encoder.to(device=device, dtype=weight_dtype)
        self._image_encoder.to(device=device, dtype=weight_dtype)
        self._vae.to(device=device, dtype=weight_dtype)
        self._unet.to(device=device, dtype=weight_dtype)
        self._ref_unet.to(device=device, dtype=weight_dtype)

        # Freeze all parameters for inference.
        self._vae.requires_grad_(False)
        self._unet.requires_grad_(False)
        self._ref_unet.requires_grad_(False)
        self._text_encoder.requires_grad_(False)
        self._image_encoder.requires_grad_(False)

        # ---------- load pose / camera data ----------
        logger.info("  Loading pose data from %s...", material_dir)
        pose_json = material_dir / "pose.json"
        with open(pose_json, "r") as f:
            metas = json.load(f)

        from torchvision.transforms import ToTensor

        totensor = ToTensor()
        cameras = []
        pose_images = []
        for lm in metas:
            cam_matrix = np.array(lm[0]).reshape(4, 4).transpose(1, 0)[:3, :4]
            cameras.append(torch.tensor(cam_matrix.reshape(-1), dtype=torch.float32))

            pose_path = material_dir / lm[1]
            if pose_path.is_file():
                pose_img = Image.open(str(pose_path))
                # CharacterGen crops pose images to (128, 0, 640, 768).
                pose_arr = np.array(pose_img.crop((128, 0, 640, 768))).astype(np.float32)
                pose_images.append(totensor(pose_arr) / 255.0)
            else:
                logger.warning("Pose image not found: %s", pose_path)

        self._camera_matrixs = torch.stack(cameras).unsqueeze(0).to(device)
        self._pose_imgs = torch.stack(pose_images).to(device) if pose_images else None

        # ---------- build inference pipeline ----------
        logger.info("  Building TuneAVideo pipeline...")
        from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline

        self._pipeline = TuneAVideoPipeline(
            vae=self._vae,
            text_encoder=self._text_encoder,
            tokenizer=self._tokenizer,
            unet=self._unet,
            scheduler=None,  # Pipeline creates its own scheduler.
        )
        # The pipeline needs to be on the correct device for inference.
        self._pipeline = self._pipeline.to(device)

        self._initialized = True
        self._device = device
        self._weight_dtype = weight_dtype
        logger.info("CharacterGen 2D stage loaded successfully.")

    def unload(self) -> None:
        """Remove all models from GPU and free VRAM."""
        from .device import unload_gpu_model

        for attr in (
            "_pipeline",
            "_vae",
            "_text_encoder",
            "_image_encoder",
            "_unet",
            "_ref_unet",
            "_pose_guider",
        ):
            unload_gpu_model(attr, self)

        self._feature_extractor = None
        self._tokenizer = None
        self._camera_matrixs = None
        self._pose_imgs = None
        self._initialized = False

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("CharacterGen 2D stage unloaded.")

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_input(
        image: Image.Image,
        target_width: int = 512,
        target_height: int = 768,
    ) -> Image.Image:
        """Preprocess an input image for CharacterGen.

        CharacterGen expects a 512x768 image with the foreground centered
        on a gray (0.5) background, similar to MV-Adapter's preprocessing
        but at a different resolution and aspect ratio.

        Steps:
          1. Convert to RGBA.
          2. Crop to the non-transparent bounding box.
          3. Resize to fit within target_width x target_height.
          4. Center-paste on a gray background canvas.

        Args:
            image: Input PIL Image (RGB or RGBA).
            target_width: Output width (default 512).
            target_height: Output height (default 768).

        Returns:
            RGBA PIL Image of size target_width x target_height.
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Find bounding box of non-transparent pixels.
        alpha_arr = np.array(image.split()[3])
        rows = np.any(alpha_arr > 0, axis=1)
        cols = np.any(alpha_arr > 0, axis=0)

        if not rows.any() or not cols.any():
            logger.warning(
                "CharacterGen prep: fully transparent image; returning gray canvas."
            )
            return Image.new("RGBA", (target_width, target_height), (128, 128, 128, 255))

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Crop with 1px margin.
        y_min = max(y_min - 1, 0)
        x_min = max(x_min - 1, 0)
        y_max = min(y_max + 2, alpha_arr.shape[0])
        x_max = min(x_max + 2, alpha_arr.shape[1])
        cropped = image.crop((x_min, y_min, x_max, y_max))

        # Resize to fit within target while preserving aspect ratio.
        cw, ch = cropped.size
        scale = min(target_width / cw, target_height / ch) * 0.9  # 90% fill
        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))
        resized = cropped.resize((new_w, new_h), Image.LANCZOS)

        # Paste centered on gray background.
        canvas = Image.new("RGBA", (target_width, target_height), (128, 128, 128, 255))
        paste_x = (target_width - new_w) // 2
        paste_y = (target_height - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y), resized)

        return canvas

    # ------------------------------------------------------------------
    # View generation
    # ------------------------------------------------------------------

    def generate_views(
        self,
        image: Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float | None = None,
        requested_view_names: list[str] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Image.Image]:
        """Generate multi-view images from a single input image.

        This method has the same signature as ``MVAdapterWrapper.generate_views``
        so the orchestrator can swap between them without code changes.

        Args:
            image: Input PIL Image (RGB or RGBA with foreground).
            num_inference_steps: Number of diffusion denoising steps.
            guidance_scale: Classifier-free guidance scale.  If None, uses
                the value from CharacterGen's config (default 5.0).
            requested_view_names: Optional subset of view names to return.
                Valid names: 'front', 'right', 'back', 'left'.
                If None, all 4 views are returned.
            batch_size: Ignored (CharacterGen always generates 4 views at
                once).  Present for API compatibility with MVAdapterWrapper.

        Returns:
            Dict keyed by view name ('front', 'right', 'back', 'left')
            mapping to PIL Images.  If ``requested_view_names`` is provided,
            only the requested views are included.
        """
        import io

        import torch
        from torchvision.utils import save_image

        if not self._initialized:
            self.load()

        if guidance_scale is None:
            guidance_scale = self._guidance_scale

        # Preprocess input image.
        prepared = self._prepare_input(image)

        # Convert to tensor for the pipeline.
        arr = np.array(prepared).astype(np.float32) / 255.0
        # Alpha-composite onto gray background for the RGB channels.
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3:4]
        composited = rgb * alpha + 0.5 * (1.0 - alpha)

        from torchvision.transforms import ToTensor

        input_tensor = ToTensor()(composited).unsqueeze(0).to(
            device=self._device, dtype=self._weight_dtype
        )

        logger.info(
            "Running CharacterGen 2D stage (steps=%d, guidance=%.1f, "
            "views=%d, resolution=%dx%d)...",
            num_inference_steps,
            guidance_scale,
            self._video_length,
            self.OUTPUT_WIDTH,
            self.OUTPUT_HEIGHT,
        )

        # Run inference.
        with torch.no_grad(), torch.autocast(
            "cuda" if self._device == "cuda" else "cpu",
            dtype=self._weight_dtype,
            enabled=self._device == "cuda",
        ):
            try:
                from einops import rearrange
            except ImportError:
                from .deps import ensure_package

                ensure_package("einops")
                from einops import rearrange

            output = self._pipeline(
                prompt="",
                image=input_tensor,
                video_length=self._video_length,
                height=self.OUTPUT_HEIGHT,
                width=self.OUTPUT_WIDTH,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                pose_image=self._pose_imgs,
                camera_matrixs=self._camera_matrixs,
                unet_condition_type=self._unet_condition_type,
                pose_guider=self._pose_guider,
                image_encoder=self._image_encoder,
                feature_extractor=self._feature_extractor,
                ref_unet=self._ref_unet,
                use_noise=self._use_noise,
                use_shifted_noise=self._use_shifted_noise,
                noise_d=self._noise_d,
            )

            # Extract individual views from video tensor.
            # Output shape: (B, C, frames, H, W).
            out = output.videos
            out = rearrange(out, "B C f H W -> (B f) C H W", f=self._video_length)

        # Convert each frame to a PIL Image.
        all_views: dict[str, Image.Image] = {}
        for i in range(min(self._video_length, out.shape[0])):
            buf = io.BytesIO()
            save_image(out[i], buf, format="PNG")
            buf.seek(0)
            view_img = Image.open(buf).copy()
            buf.close()

            if i < len(self.VIEW_ORDER):
                view_name = self.VIEW_ORDER[i]
                all_views[view_name] = view_img

        del output, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Filter to requested views if specified.
        if requested_view_names is not None:
            valid_names = set(self.VIEW_ORDER)
            for name in requested_view_names:
                if name not in valid_names:
                    raise ValueError(
                        f"Unknown CharacterGen view '{name}'. "
                        f"Valid names: {', '.join(sorted(valid_names))}"
                    )
            all_views = {k: v for k, v in all_views.items() if k in requested_view_names}

        logger.info(
            "CharacterGen generated %d views: %s",
            len(all_views),
            list(all_views.keys()),
        )
        return all_views
