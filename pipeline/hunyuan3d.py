"""
Shape generation using the official Hunyuan3D-2.1 fp16 checkpoint.

This project uses the Hunyuan3D-2.1 shape generator only:
    1. Load the official `hunyuan3d-dit-v2-1` checkpoint
    2. Run geometry generation in fp16 on CUDA when available
    3. Export a geometry-only mesh/GLB with no texture stage

Reference: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
Model: https://huggingface.co/tencent/Hunyuan3D-2.1
"""

from contextlib import nullcontext
from importlib import import_module
import logging
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Official Hugging Face repo and the fp16 shape subfolder we target by default.
HUNYUAN3D_REPO_ID = "tencent/Hunyuan3D-2.1"
HUNYUAN3D_SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-1"
HUNYUAN3D_REPO_DIRNAME = "Hunyuan3D-2.1"

# Multi-view variant for use with Zero123++ views
HUNYUAN3D_2MV_REPO_ID = "tencent/Hunyuan3D-2mv"
HUNYUAN3D_2MV_SUBFOLDER = "hunyuan3d-dit-v2-mv"


class Hunyuan3DWrapper:
    """
    Wrapper around Hunyuan3D-2.1 for geometry generation only.

    The default profile is tuned for cards like the RTX 3080 Ti 16GB:
      - official Hunyuan3D-2.1 shape checkpoint
      - fp16 on CUDA
      - no texture/PBR generation
    """

    def __init__(self, device_config, model_path: str | None = None, variant: str = "single"):
        """
        Initialize Hunyuan3D-2.1 wrapper.

        Args:
            device_config: DeviceConfig from device.py.
            model_path: Optional local path or Hugging Face repo ID. If None,
                the official checkpoint is used (varies by variant).
            variant: ``"single"`` for the default single-image shape pipeline,
                ``"multiview"`` for the Hunyuan3D-2mv multi-view shape pipeline.
        """
        self.device_config = device_config
        self.model_path = model_path
        self.variant = variant
        self.pipeline = None
        self._initialized = False

    def _candidate_code_roots(self) -> list[Path]:
        """Return likely locations of a local Hunyuan3D-2.1 repo checkout."""
        candidates: list[Path] = []

        env_path = os.environ.get("HUNYUAN3D_21_REPO_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())

        if self.model_path:
            model_dir = Path(self.model_path)
            candidates.append(model_dir)
            candidates.append(model_dir.parent)

        this_file = Path(__file__).resolve()
        candidates.extend([
            Path.cwd() / HUNYUAN3D_REPO_DIRNAME,
            this_file.parents[1] / HUNYUAN3D_REPO_DIRNAME,
            this_file.parents[2] / HUNYUAN3D_REPO_DIRNAME,
        ])

        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                unique_candidates.append(candidate)
                seen.add(key)
        return unique_candidates

    def _prepare_hunyuan_import(self) -> bool:
        """
        Add a local Hunyuan3D-2.1 checkout to ``sys.path`` when available.

        The official 2.1 repository keeps the importable package inside
        `<repo>/hy3dshape`, so we add that directory rather than the repo root.
        """
        for repo_root in self._candidate_code_roots():
            package_root = repo_root / "hy3dshape"
            pipeline_file = package_root / "hy3dshape" / "pipelines.py"
            if pipeline_file.exists():
                package_root_str = str(package_root)
                if package_root_str not in sys.path:
                    sys.path.insert(0, package_root_str)
                logger.info(f"Using local Hunyuan3D-2.1 code checkout: {repo_root}")
                return True
        return False

    def _import_shape_pipeline(self):
        """Import the Hunyuan shape pipeline from the best available source."""
        self._prepare_hunyuan_import()

        try:
            module = import_module("hy3dshape.pipelines")
            logger.info("Imported Hunyuan3D-2.1 shape pipeline from hy3dshape.")
            return module.Hunyuan3DDiTFlowMatchingPipeline
        except ImportError:
            logger.debug("hy3dshape import unavailable, trying hy3dgen fallback.")

        try:
            module = import_module("hy3dgen.shapegen")
            logger.info("Imported Hunyuan shape pipeline from hy3dgen.")
            return module.Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as exc:
            searched = "\n".join(f"  - {path}" for path in self._candidate_code_roots())
            raise ImportError(
                "Could not import the Hunyuan3D-2.1 shape pipeline.\n"
                "Expected either:\n"
                "  1. A local Hunyuan3D-2.1 repo checkout (with `hy3dshape`) in one of:\n"
                f"{searched}\n"
                "  2. An installed legacy `hy3dgen` package.\n"
                "To fix this, clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 "
                "next to this project and install the shape-side Python requirements."
            ) from exc

    def resolve_model_source(self) -> tuple[str, str | None]:
        """
        Resolve the model source and optional subfolder.

        Returns:
            Tuple of (model source, optional subfolder).
        """
        if not self.model_path:
            if self.variant == "multiview":
                logger.info(
                    f"Using official Hunyuan3D-2mv model: "
                    f"{HUNYUAN3D_2MV_REPO_ID}/{HUNYUAN3D_2MV_SUBFOLDER}"
                )
                return HUNYUAN3D_2MV_REPO_ID, HUNYUAN3D_2MV_SUBFOLDER
            logger.info(
                "Using official Hunyuan3D-2.1 fp16 shape model: "
                f"{HUNYUAN3D_REPO_ID}/{HUNYUAN3D_SHAPE_SUBFOLDER}"
            )
            return HUNYUAN3D_REPO_ID, HUNYUAN3D_SHAPE_SUBFOLDER

        model_dir = Path(self.model_path)
        if model_dir.exists():
            if (model_dir / "config.yaml").exists():
                logger.info(f"Using local Hunyuan3D shape checkpoint: {model_dir}")
                return str(model_dir), None

            if (model_dir / HUNYUAN3D_SHAPE_SUBFOLDER / "config.yaml").exists():
                logger.info(f"Using local Hunyuan3D repo checkout: {model_dir}")
                return str(model_dir), HUNYUAN3D_SHAPE_SUBFOLDER

            logger.warning(
                "Custom Hunyuan3D model path does not match the expected repo layout. "
                "Attempting to load it as a repo root with the shape subfolder."
            )
            return str(model_dir), HUNYUAN3D_SHAPE_SUBFOLDER

        logger.info(
            "Treating configured Hunyuan model path as a Hugging Face repo ID: "
            f"{self.model_path}"
        )
        return self.model_path, HUNYUAN3D_SHAPE_SUBFOLDER

    def initialize(self) -> None:
        """
        Load the Hunyuan3D-2.1 shape model for inference.

        This is called lazily on first generate() call, or can be called
        explicitly to pre-load models.
        """
        if self._initialized:
            return

        logger.info("Initializing Hunyuan3D-2.1 shape pipeline...")
        logger.info(f"  Device: {self.device_config.device}")
        logger.info(f"  Dtype: {self.device_config.dtype}")

        if not self.device_config.has_gpu:
            logger.warning(
                "No GPU detected. Hunyuan3D-2.1 shape generation will run on CPU "
                "with float32. This will be much slower than fp16 on CUDA."
            )

        try:
            model_source, subfolder = self.resolve_model_source()
            self._load_pipeline(model_source, subfolder)
            self._initialized = True
            logger.info("Hunyuan3D-2.1 initialized successfully.")
        except ImportError as e:
            logger.error(
                f"Hunyuan3D-2.1 dependencies not found: {e}\n"
                "Please install Hunyuan3D-2.1 following the instructions:\n"
                "  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git\n"
                "  # clone it next to this project or set HUNYUAN3D_21_REPO_PATH\n"
                "  pip install einops omegaconf pytorch-lightning torchdiffeq\n"
                "Or install the full official requirements from the repo:\n"
                "  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git\n"
                "  cd Hunyuan3D-2.1 && pip install -r requirements.txt"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Hunyuan3D-2.1: {e}")
            raise

    def _load_pipeline(self, model_source: str, subfolder: str | None) -> None:
        """
        Load the official Hunyuan3D-2.1 shape pipeline.

        Args:
            model_source: Hugging Face repo ID or local model directory.
            subfolder: Optional subfolder containing the shape checkpoint.
        """
        import torch

        device = str(self.device_config.device)
        dtype = self.device_config.dtype

        # If on CPU, force float32 (half precision is not supported on CPU)
        if not self.device_config.has_gpu:
            dtype = torch.float32

        Hunyuan3DDiTFlowMatchingPipeline = self._import_shape_pipeline()

        logger.info("Loading Hunyuan3D-2.1 fp16 shape model...")
        load_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        if subfolder is not None:
            load_kwargs["subfolder"] = subfolder

        try:
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_source,
                **load_kwargs,
            )
        except TypeError:
            # Older loaders may not accept every kwarg.
            load_kwargs.pop("dtype", None)
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_source,
                **load_kwargs,
            )

        # Keep the internal model in fp16 on CUDA when the API exposes it.
        if hasattr(self.pipeline, "to"):
            self.pipeline.to(device=device, dtype=dtype)

    def generate(
        self,
        image: Image.Image,
        seed: int = 42,
        guidance_scale: float = 7.5,
        num_steps: int = 50,
    ) -> dict:
        """
        Run Hunyuan3D-2.1 shape generation.

        Args:
            image: Preprocessed PIL Image (RGB, background removed).
            seed: Random seed for reproducibility.
            guidance_scale: Classifier-free guidance scale.
            num_steps: Number of diffusion steps.

        Returns:
            Dictionary with keys:
                - "mesh": trimesh.Trimesh object with the generated geometry
                - texture/PBR keys are always None in this shape-only profile
        """
        import torch

        self.initialize()

        logger.info("Running Hunyuan3D-2.1 shape generation (geometry only)...")
        logger.info(
            f"  Seed: {seed}, Guidance: {guidance_scale}, Steps: {num_steps}"
        )

        if not self.device_config.has_gpu:
            logger.warning(
                "Running on CPU — shape generation may take 30-60 minutes."
            )

        generator = torch.Generator(device=str(self.device_config.device))
        generator.manual_seed(seed)

        try:
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.device_config.dtype)
                if self.device_config.has_gpu
                else nullcontext()
            )

            with torch.no_grad():
                with autocast_ctx:
                    outputs = self.pipeline(
                        image=image,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        generator=generator,
                        enable_pbar=False,
                    )

            result = self._extract_results(outputs)

            mesh = result["mesh"]
            logger.info(
                f"Hunyuan3D-2.1 generation complete. "
                f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}"
            )

            return result

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "GPU out of memory during Hunyuan3D-2.1 generation. "
                "Try: 1) Closing other GPU applications, "
                "2) Using --force-cpu flag, "
                "3) Using a GPU with more VRAM (10GB+ required, 16GB recommended)."
            )
            raise
        except Exception as e:
            logger.error(f"Hunyuan3D-2.1 generation failed: {e}")
            raise

    def load(self):
        """Load the shape model (alias for initialize)."""
        self.initialize()

    def unload(self):
        """Remove shape model from GPU and free VRAM."""
        from .device import unload_gpu_model

        unload_gpu_model("pipeline", self)
        self._initialized = False
        logger.info("Hunyuan3D shape pipeline unloaded.")

    def generate_from_multiview(
        self,
        views: dict,
        seed: int = 42,
        guidance_scale: float = 5.0,
        num_steps: int = 50,
        octree_resolution: int = 384,
    ) -> dict:
        """Generate mesh from multiple views (for Hunyuan3D-2mv).

        Args:
            views: Dict with keys ``front``, ``left``, ``back``, ``right`` as
                PIL Images.
            seed: Random seed for reproducibility.
            guidance_scale: Classifier-free guidance scale.
            num_steps: Number of diffusion steps.
            octree_resolution: Octree resolution for mesh extraction.

        Returns:
            Dict with ``mesh`` key containing a ``trimesh.Trimesh``.
        """
        import torch

        self.initialize()

        logger.info(
            f"Running Hunyuan3D-2mv (views={list(views.keys())}, "
            f"steps={num_steps}, octree={octree_resolution})..."
        )

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # The 2mv pipeline accepts a dict of images for multi-view input.
        # MVImageProcessorV2 handles the dict with keys front/left/back/right.
        with torch.inference_mode():
            outputs = self.pipeline(
                image=views,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
                num_chunks=20000,  # Higher for 512 octree safety
                output_type="trimesh",
                generator=generator,
                enable_pbar=False,
            )

        return self._extract_results(outputs)

    def _extract_results(self, outputs) -> dict:
        """
        Extract the generated mesh from pipeline outputs.

        Handles various output formats from different pipeline versions.

        Args:
            outputs: Raw pipeline output.

        Returns:
            Dictionary with mesh and placeholder texture keys.
        """
        import trimesh

        result = {
            "mesh": None,
            "texture": None,
            "normal_map": None,
            "metallic_map": None,
            "roughness_map": None,
        }

        payload = outputs
        while isinstance(payload, (list, tuple)) and payload:
            payload = payload[0]

        if isinstance(payload, trimesh.Trimesh):
            result["mesh"] = payload
            return result

        # Extract mesh
        if isinstance(outputs, trimesh.Trimesh):
            result["mesh"] = outputs
            return result

        if hasattr(payload, "mesh"):
            mesh_data = payload.mesh
            if isinstance(mesh_data, trimesh.Trimesh):
                result["mesh"] = mesh_data
            elif hasattr(mesh_data, "vertices") and hasattr(mesh_data, "faces"):
                result["mesh"] = trimesh.Trimesh(
                    vertices=np.array(mesh_data.vertices),
                    faces=np.array(mesh_data.faces),
                )

        if hasattr(payload, "meshes") and payload.meshes:
            mesh_data = payload.meshes[0]
            if isinstance(mesh_data, trimesh.Trimesh):
                result["mesh"] = mesh_data
            elif hasattr(mesh_data, "vertices") and hasattr(mesh_data, "faces"):
                result["mesh"] = trimesh.Trimesh(
                    vertices=np.array(mesh_data.vertices),
                    faces=np.array(mesh_data.faces),
                )

        if isinstance(payload, dict):
            if "mesh" in payload:
                mesh_data = payload["mesh"]
                if isinstance(mesh_data, trimesh.Trimesh):
                    result["mesh"] = mesh_data
                else:
                    result["mesh"] = trimesh.Trimesh(
                        vertices=np.array(mesh_data["vertices"]),
                        faces=np.array(mesh_data["faces"]),
                    )
            elif "vertices" in payload and "faces" in payload:
                result["mesh"] = trimesh.Trimesh(
                    vertices=np.array(payload["vertices"]),
                    faces=np.array(payload["faces"]),
                )

        if result["mesh"] is None:
            raise ValueError(
                f"Could not extract mesh from Hunyuan3D-2.1 output of type "
                f"{type(payload)}. This may indicate an incompatible version."
            )

        return result
