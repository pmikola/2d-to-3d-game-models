"""
Geometry generation using TripoSG (VAST-AI, MIT License).

TripoSG is a 1.5B parameter rectified flow model for single-image-to-3D.
Requires ~8GB VRAM — fits comfortably on a T4 GPU (16GB).

Reference: https://github.com/VAST-AI-Research/TripoSG
Model: https://huggingface.co/VAST-AI/TripoSG
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TripoSGWrapper:
    """
    Wrapper around TripoSG for mesh generation from 2D images.

    TripoSG uses a rectified flow transformer to generate 3D geometry
    from a single image. It includes its own background removal (BRIA RMBG-1.4).
    """

    def __init__(self, device_config, repo_path: str | None = None):
        """
        Initialize TripoSG wrapper.

        Args:
            device_config: DeviceConfig from device.py.
            repo_path: Path to cloned TripoSG repo. If None, attempts auto-download.
        """
        self.device_config = device_config
        self.repo_path = repo_path or "/content/TripoSG"
        self.pipeline = None
        self.rmbg_net = None
        self._initialized = False

    def initialize(self) -> None:
        """Download weights and load models."""
        if self._initialized:
            return

        import torch

        logger.info("Initializing TripoSG pipeline...")
        logger.info(f"  Device: {self.device_config.device}")

        # Clone repo if needed
        repo = Path(self.repo_path)
        if not repo.exists():
            logger.info("Cloning TripoSG repository...")
            import subprocess
            subprocess.check_call([
                "git", "clone",
                "https://github.com/VAST-AI-Research/TripoSG.git",
                str(repo),
            ])

        # Add to path
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "scripts"))

        # Download model weights
        from huggingface_hub import snapshot_download

        weights_dir = repo / "pretrained_weights"
        weights_dir.mkdir(exist_ok=True)

        triposg_dir = str(weights_dir / "TripoSG")
        if not Path(triposg_dir).exists():
            logger.info("Downloading TripoSG weights (~3GB)...")
            triposg_dir = snapshot_download(
                repo_id="VAST-AI/TripoSG",
                local_dir=triposg_dir,
            )

        rmbg_dir = str(weights_dir / "RMBG-1.4")
        if not Path(rmbg_dir).exists():
            logger.info("Downloading RMBG-1.4 weights...")
            rmbg_dir = snapshot_download(
                repo_id="briaai/RMBG-1.4",
                local_dir=rmbg_dir,
            )

        # Load models
        device = str(self.device_config.device)
        dtype = self.device_config.dtype

        try:
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline
            from scripts.briarmbg import BriaRMBG

            logger.info("Loading RMBG-1.4 (background removal)...")
            self.rmbg_net = BriaRMBG.from_pretrained(rmbg_dir).to(device)
            self.rmbg_net.eval()

            logger.info("Loading TripoSG pipeline...")
            self.pipeline = TripoSGPipeline.from_pretrained(triposg_dir)
            self.pipeline.to(device, dtype)

            self._initialized = True
            logger.info("TripoSG initialized successfully.")

        except ImportError as e:
            logger.error(
                f"TripoSG import failed: {e}\n"
                "Try installing dependencies:\n"
                "  pip install diso --no-build-isolation\n"
                "  pip install diffusers transformers einops trimesh omegaconf peft"
            )
            raise

    def generate_mesh(
        self,
        image: Image.Image,
        seed: int = 42,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 50,
    ):
        """
        Generate 3D mesh from a preprocessed image.

        Args:
            image: PIL Image (RGB). Background removal is done internally.
            seed: Random seed for reproducibility.
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Number of diffusion steps.

        Returns:
            trimesh.Trimesh object.
        """
        import torch
        import trimesh

        self.initialize()

        logger.info("Generating 3D geometry with TripoSG...")
        logger.info(f"  Seed: {seed}, Guidance: {guidance_scale}, Steps: {num_inference_steps}")

        device = str(self.device_config.device)

        # Preprocess image (background removal)
        try:
            from scripts.image_process import prepare_image

            img_processed = prepare_image(
                image,
                bg_color=np.array([1.0, 1.0, 1.0]),
                rmbg_net=self.rmbg_net,
            )
        except ImportError:
            logger.warning("TripoSG image_process not available, using image as-is")
            img_processed = image

        # Run inference
        with torch.no_grad():
            outputs = self.pipeline(
                image=img_processed,
                generator=torch.Generator(device=device).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).samples[0]

        # Convert to trimesh
        vertices = outputs[0].astype(np.float32)
        faces = np.ascontiguousarray(outputs[1])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        logger.info(
            f"TripoSG generation complete. "
            f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}"
        )
        return mesh
