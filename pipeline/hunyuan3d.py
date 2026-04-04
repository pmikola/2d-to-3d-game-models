"""
Integrated geometry + PBR texturing using Hunyuan3D-2.1 (Tencent, Apache 2.0).

Hunyuan3D-2.1 provides BOTH geometry AND PBR texturing in a single integrated
system, replacing the separate Hi3DGen (geometry) + Text2Tex (texturing) stages.

Architecture:
    1. DiT-based geometry backbone generates 3D mesh from a single image
    2. Multi-view PBR diffusion painter produces albedo, metallic, and roughness maps
    3. Output: textured mesh with full PBR material

Reference: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 (Apache 2.0)
Model: https://huggingface.co/tencent/Hunyuan3D-2
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default HuggingFace repo for model weights
HUNYUAN3D_REPO_ID = "tencent/Hunyuan3D-2"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hunyuan3d"


class Hunyuan3DWrapper:
    """
    Wrapper around Hunyuan3D-2.1 for combined mesh generation and PBR texturing.

    Hunyuan3D-2.1 uses:
      - A DiT-based flow-matching pipeline for geometry generation
      - A multi-view PBR diffusion painter for albedo, metallic, and roughness maps
      - Single integrated pipeline replacing separate geometry + texturing stages
    """

    def __init__(self, device_config, model_path: str | None = None):
        """
        Initialize Hunyuan3D-2.1 wrapper.

        Args:
            device_config: DeviceConfig from device.py.
            model_path: Path to local model weights. If None, downloads from
                HuggingFace to ~/.cache/hunyuan3d/ on first run.
        """
        self.device_config = device_config
        self.model_path = model_path
        self.pipeline = None
        self._initialized = False

    def check_and_download_model(self) -> str:
        """
        Check if Hunyuan3D-2.1 weights exist locally; download if not.

        Returns:
            Path to the local model directory.
        """
        if self.model_path:
            model_dir = Path(self.model_path)
            if model_dir.exists():
                logger.info(f"Using user-provided model path: {model_dir}")
                return str(model_dir)
            else:
                logger.warning(
                    f"Provided model path {model_dir} does not exist. "
                    "Falling back to default cache directory."
                )

        model_dir = DEFAULT_CACHE_DIR
        if not (model_dir / "config.json").exists():
            logger.info("Hunyuan3D-2.1 weights not found locally. Downloading...")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=HUNYUAN3D_REPO_ID,
                local_dir=str(model_dir),
            )
            logger.info(f"Downloaded Hunyuan3D-2.1 to {model_dir}")
        else:
            logger.info(f"Found Hunyuan3D-2.1 weights at {model_dir}")

        return str(model_dir)

    def initialize(self) -> None:
        """
        Download (if needed) and load Hunyuan3D-2.1 models for inference.

        This is called lazily on first generate() call, or can be called
        explicitly to pre-load models.
        """
        if self._initialized:
            return

        logger.info("Initializing Hunyuan3D-2.1 pipeline...")
        logger.info(f"  Device: {self.device_config.device}")
        logger.info(f"  Dtype: {self.device_config.dtype}")

        if not self.device_config.has_gpu:
            logger.warning(
                "No GPU detected. Hunyuan3D-2.1 will run on CPU with float32. "
                "This will be extremely slow (30+ minutes per image)."
            )

        try:
            model_dir = self.check_and_download_model()
            self._load_pipeline(model_dir)
            self._initialized = True
            logger.info("Hunyuan3D-2.1 initialized successfully.")
        except ImportError as e:
            logger.error(
                f"Hunyuan3D-2.1 dependencies not found: {e}\n"
                "Please install Hunyuan3D-2.1 following the instructions:\n"
                "  pip install hy3dgen\n"
                "Or clone the repo:\n"
                "  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git\n"
                "  cd Hunyuan3D-2.1 && pip install -r requirements.txt"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Hunyuan3D-2.1: {e}")
            raise

    def _load_pipeline(self, model_dir: str) -> None:
        """
        Load the Hunyuan3D-2.1 pipeline, trying multiple import paths.

        Args:
            model_dir: Path to local model weights directory.
        """
        import torch

        device = str(self.device_config.device)
        dtype = self.device_config.dtype

        # If on CPU, force float32 (half precision is not supported on CPU)
        if not self.device_config.has_gpu:
            dtype = torch.float32

        # Try 1: Official hy3dgen package (preferred)
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            logger.info("Loading Hunyuan3D-2.1 via hy3dgen package...")
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_dir,
                device=device,
                torch_dtype=dtype,
            )
            logger.info("Loaded Hunyuan3D-2.1 pipeline from hy3dgen.")
            return
        except ImportError:
            logger.debug("hy3dgen package not found, trying diffusers integration...")

        # Try 2: diffusers integration (if/when Hunyuan3D is merged into diffusers)
        try:
            from diffusers import HunyuanDiTPipeline

            logger.info("Loading Hunyuan3D-2.1 via diffusers...")
            self.pipeline = HunyuanDiTPipeline.from_pretrained(
                model_dir,
                torch_dtype=dtype,
            )
            self.pipeline.to(device)
            logger.info("Loaded Hunyuan3D-2.1 pipeline from diffusers.")
            return
        except ImportError:
            logger.debug("diffusers HunyuanDiTPipeline not available.")

        # Neither import path worked
        raise ImportError(
            "Could not import Hunyuan3D-2.1 pipeline. Install one of:\n"
            "  1. hy3dgen (official): pip install hy3dgen\n"
            "  2. Clone the repo: git clone "
            "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git && "
            "cd Hunyuan3D-2.1 && pip install -r requirements.txt\n"
            "  3. diffusers (when available): pip install diffusers>=0.30\n"
            "See https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 for details."
        )

    def generate(
        self,
        image: Image.Image,
        seed: int = 42,
        guidance_scale: float = 7.5,
        num_steps: int = 50,
    ) -> dict:
        """
        Run the full Hunyuan3D-2.1 pipeline: geometry generation + PBR texturing.

        Args:
            image: Preprocessed PIL Image (RGB, background removed).
            seed: Random seed for reproducibility.
            guidance_scale: Classifier-free guidance scale.
            num_steps: Number of diffusion steps.

        Returns:
            Dictionary with keys:
                - "mesh": trimesh.Trimesh object with the generated geometry
                - "texture": PIL Image (albedo/diffuse map) or None
                - "normal_map": PIL Image (normal map) or None
                - "metallic_map": PIL Image (metallic map) or None
                - "roughness_map": PIL Image (roughness map) or None
        """
        import torch
        import trimesh

        self.initialize()

        logger.info("Running Hunyuan3D-2.1 full pipeline (geometry + PBR texturing)...")
        logger.info(
            f"  Seed: {seed}, Guidance: {guidance_scale}, Steps: {num_steps}"
        )

        if not self.device_config.has_gpu:
            logger.warning(
                "Running on CPU — geometry + texturing may take 30-60 minutes."
            )

        # Set random seed
        torch.manual_seed(seed)
        if self.device_config.has_gpu:
            torch.cuda.manual_seed(seed)

        try:
            with torch.no_grad():
                if self.device_config.has_gpu:
                    with torch.cuda.amp.autocast(dtype=self.device_config.dtype):
                        outputs = self.pipeline(
                            image,
                            seed=seed,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_steps,
                        )
                else:
                    outputs = self.pipeline(
                        image,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                    )

            result = self._extract_results(outputs)

            mesh = result["mesh"]
            logger.info(
                f"Hunyuan3D-2.1 generation complete. "
                f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}"
            )

            has_texture = result["texture"] is not None
            has_pbr = result["metallic_map"] is not None
            logger.info(
                f"  Texture: {'yes' if has_texture else 'no'}, "
                f"PBR maps: {'yes' if has_pbr else 'no'}"
            )

            return result

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "GPU out of memory during Hunyuan3D-2.1 generation. "
                "Try: 1) Closing other GPU applications, "
                "2) Using --force-cpu flag, "
                "3) Using a GPU with more VRAM (16GB+ recommended)."
            )
            raise
        except Exception as e:
            logger.error(f"Hunyuan3D-2.1 generation failed: {e}")
            raise

    def _extract_results(self, outputs) -> dict:
        """
        Extract mesh and PBR texture maps from pipeline outputs.

        Handles various output formats from different pipeline versions.

        Args:
            outputs: Raw pipeline output.

        Returns:
            Dictionary with mesh and optional texture maps.
        """
        import trimesh

        result = {
            "mesh": None,
            "texture": None,
            "normal_map": None,
            "metallic_map": None,
            "roughness_map": None,
        }

        # Extract mesh
        if isinstance(outputs, trimesh.Trimesh):
            result["mesh"] = outputs
            return result

        if hasattr(outputs, "mesh"):
            mesh_data = outputs.mesh
            if isinstance(mesh_data, trimesh.Trimesh):
                result["mesh"] = mesh_data
            elif hasattr(mesh_data, "vertices") and hasattr(mesh_data, "faces"):
                result["mesh"] = trimesh.Trimesh(
                    vertices=np.array(mesh_data.vertices),
                    faces=np.array(mesh_data.faces),
                )

        if hasattr(outputs, "meshes") and outputs.meshes:
            mesh_data = outputs.meshes[0]
            if isinstance(mesh_data, trimesh.Trimesh):
                result["mesh"] = mesh_data
            elif hasattr(mesh_data, "vertices") and hasattr(mesh_data, "faces"):
                result["mesh"] = trimesh.Trimesh(
                    vertices=np.array(mesh_data.vertices),
                    faces=np.array(mesh_data.faces),
                )

        if isinstance(outputs, dict):
            if "mesh" in outputs:
                mesh_data = outputs["mesh"]
                if isinstance(mesh_data, trimesh.Trimesh):
                    result["mesh"] = mesh_data
                else:
                    result["mesh"] = trimesh.Trimesh(
                        vertices=np.array(mesh_data["vertices"]),
                        faces=np.array(mesh_data["faces"]),
                    )
            elif "vertices" in outputs and "faces" in outputs:
                result["mesh"] = trimesh.Trimesh(
                    vertices=np.array(outputs["vertices"]),
                    faces=np.array(outputs["faces"]),
                )

        if result["mesh"] is None:
            raise ValueError(
                f"Could not extract mesh from Hunyuan3D-2.1 output of type "
                f"{type(outputs)}. This may indicate an incompatible version."
            )

        # Extract PBR texture maps (Hunyuan3D-Paint output)
        texture_keys = {
            "texture": ["texture", "albedo", "diffuse", "texture_map", "albedo_map"],
            "normal_map": ["normal_map", "normal", "normals"],
            "metallic_map": ["metallic_map", "metallic", "metalness"],
            "roughness_map": ["roughness_map", "roughness"],
        }

        source = outputs if isinstance(outputs, dict) else outputs

        for result_key, candidate_names in texture_keys.items():
            for name in candidate_names:
                tex = None
                if isinstance(source, dict) and name in source:
                    tex = source[name]
                elif hasattr(source, name):
                    tex = getattr(source, name)

                if tex is not None:
                    if isinstance(tex, Image.Image):
                        result[result_key] = tex
                    elif isinstance(tex, np.ndarray):
                        result[result_key] = Image.fromarray(
                            (tex * 255).clip(0, 255).astype(np.uint8)
                            if tex.dtype in (np.float32, np.float64)
                            else tex
                        )
                    break

        return result
