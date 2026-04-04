"""
Geometry generation using Hi3DGen (ICCV 2025, ByteDance).

Hi3DGen is the current SOTA for open-source image-to-3D geometry generation,
outperforming TRELLIS, TripoSG, Hunyuan3D, and CraftsMan in benchmarks.

Pipeline:
    1. NiRNE: Image -> high-quality normal map
    2. NoRLD: Normal map -> 3D geometry via TRELLIS-based backbone
    3. Output: trimesh mesh object

Reference: https://github.com/bytedance/Hi3DGen (MIT License)
Model: https://huggingface.co/Stable-X/trellis-normal-v0-1
"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Hi3DGenWrapper:
    """
    Wrapper around Hi3DGen for mesh generation from 2D images.

    Hi3DGen uses:
      - NiRNE (Noise-injected Regressive Normal Estimator) for normal prediction
      - NoRLD (Normal-Regularized Latent Diffusion) for geometry generation
      - TRELLIS-based backbone for 3D structure
    """

    def __init__(self, device_config, hi3dgen_path: str | None = None):
        """
        Initialize Hi3DGen wrapper.

        Args:
            device_config: DeviceConfig from device.py.
            hi3dgen_path: Path to cloned Hi3DGen repo. If None, attempts auto-setup.
        """
        self.device_config = device_config
        self.hi3dgen_path = hi3dgen_path
        self.pipeline = None
        self._initialized = False

    def initialize(self) -> None:
        """
        Load Hi3DGen models and prepare for inference.

        This downloads model weights from HuggingFace on first run.
        """
        if self._initialized:
            return

        logger.info("Initializing Hi3DGen geometry pipeline...")
        logger.info(f"  Device: {self.device_config.device}")
        logger.info(f"  Dtype: {self.device_config.dtype}")

        try:
            self._load_hi3dgen()
            self._initialized = True
            logger.info("Hi3DGen initialized successfully.")
        except ImportError as e:
            logger.error(
                f"Hi3DGen dependencies not found: {e}\n"
                "Please install Hi3DGen following the instructions in README.md:\n"
                "  git clone https://github.com/bytedance/Hi3DGen.git\n"
                "  cd Hi3DGen && pip install -r requirements.txt"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Hi3DGen: {e}")
            raise

    def _load_hi3dgen(self) -> None:
        """Load Hi3DGen pipeline from the installed package or local repo."""
        import torch

        # Add Hi3DGen to path if a local repo path is provided
        if self.hi3dgen_path:
            hi3dgen_dir = Path(self.hi3dgen_path)
            if hi3dgen_dir.exists():
                sys.path.insert(0, str(hi3dgen_dir))
                logger.info(f"Added Hi3DGen path: {hi3dgen_dir}")

        # Try importing Hi3DGen's core pipeline
        # Hi3DGen uses TRELLIS internally for geometry diffusion
        try:
            from hi3dgen.pipelines import Hi3DGenPipeline

            self.pipeline = Hi3DGenPipeline.from_pretrained(
                "Stable-X/trellis-normal-v0-1",
                device=str(self.device_config.device),
                torch_dtype=self.device_config.dtype,
            )
            logger.info("Loaded Hi3DGen pipeline from installed package.")
            return
        except ImportError:
            pass

        # Fallback: try loading TRELLIS directly (Hi3DGen builds on TRELLIS)
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline

            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "Stable-X/trellis-normal-v0-1"
            )
            self.pipeline.to(self.device_config.device)
            logger.info("Loaded TRELLIS pipeline as fallback.")
            return
        except ImportError:
            pass

        # If neither is available, raise a helpful error
        raise ImportError(
            "Neither Hi3DGen nor TRELLIS pipeline could be imported. "
            "Please install one of:\n"
            "  1. Hi3DGen: git clone https://github.com/bytedance/Hi3DGen.git && "
            "cd Hi3DGen && pip install -r requirements.txt\n"
            "  2. TRELLIS: pip install trellis-3d\n"
            "See README.md for detailed installation instructions."
        )

    def generate_mesh(
        self,
        image: Image.Image,
        seed: int = 42,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ):
        """
        Generate 3D mesh from a preprocessed image.

        Args:
            image: Preprocessed PIL Image (512x512, RGB, background removed).
            seed: Random seed for reproducibility.
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Number of diffusion steps.

        Returns:
            trimesh.Trimesh object with the generated geometry.
        """
        import torch
        import trimesh

        self.initialize()

        logger.info("Generating 3D geometry from image...")
        logger.info(f"  Seed: {seed}, Guidance: {guidance_scale}, Steps: {num_inference_steps}")

        if not self.device_config.has_gpu:
            logger.warning(
                "Running geometry generation on CPU — this may take 10-30 minutes."
            )

        # Set random seed
        torch.manual_seed(seed)
        if self.device_config.has_gpu:
            torch.cuda.manual_seed(seed)

        try:
            # Run the Hi3DGen / TRELLIS pipeline
            with torch.no_grad():
                if self.device_config.has_gpu:
                    with torch.autocast(device_type="cuda", dtype=self.device_config.dtype):
                        outputs = self.pipeline(
                            image,
                            seed=seed,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                        )
                else:
                    outputs = self.pipeline(
                        image,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                    )

            # Extract mesh from pipeline output
            mesh = self._extract_mesh(outputs)

            logger.info(
                f"Geometry generation complete. "
                f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}"
            )
            return mesh

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "GPU out of memory during geometry generation. "
                "Try: 1) Closing other GPU applications, "
                "2) Using --force-cpu flag, "
                "3) Using a GPU with more VRAM (16GB+ recommended)."
            )
            raise
        except Exception as e:
            logger.error(f"Geometry generation failed: {e}")
            raise

    def _extract_mesh(self, outputs):
        """Extract trimesh object from pipeline outputs."""
        import trimesh

        # Hi3DGen / TRELLIS outputs vary by version
        # Handle different output formats
        if isinstance(outputs, trimesh.Trimesh):
            return outputs

        if hasattr(outputs, "mesh"):
            return outputs.mesh

        if hasattr(outputs, "meshes") and len(outputs.meshes) > 0:
            return outputs.meshes[0]

        # If output contains vertices and faces directly
        if hasattr(outputs, "vertices") and hasattr(outputs, "faces"):
            return trimesh.Trimesh(
                vertices=np.array(outputs.vertices),
                faces=np.array(outputs.faces),
            )

        # Try to extract from dict-like output
        if isinstance(outputs, dict):
            if "mesh" in outputs:
                return outputs["mesh"]
            if "vertices" in outputs and "faces" in outputs:
                return trimesh.Trimesh(
                    vertices=np.array(outputs["vertices"]),
                    faces=np.array(outputs["faces"]),
                )

        raise ValueError(
            f"Could not extract mesh from pipeline output of type {type(outputs)}. "
            "This may indicate an incompatible Hi3DGen version."
        )


def unwrap_uvs(mesh) -> None:
    """
    Apply UV unwrapping to a mesh using xatlas.

    Text2Tex/TEXTure require UV-mapped meshes. Hi3DGen output typically
    does not include UV coordinates, so we generate them here.

    Args:
        mesh: trimesh.Trimesh object (modified in-place).
    """
    try:
        import xatlas

        logger.info("UV unwrapping mesh with xatlas...")

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.uint32)

        # Run xatlas UV unwrapping
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, faces)
        atlas.generate()

        # Get the unwrapped mesh data
        vmapping, new_faces, uvs = atlas[0]

        # Update mesh with UV data
        mesh.vertices = vertices[vmapping]
        mesh.faces = new_faces
        try:
            import trimesh

            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        except Exception as e:
            logger.warning(f"Could not set UV visual via TextureVisuals: {e}")
            # Fallback: try setting UVs directly
            mesh.visual = mesh.visual.__class__(uv=uvs, material=None)

        logger.info(
            f"UV unwrapping complete. "
            f"Vertices: {len(mesh.vertices)}, UVs: {len(uvs)}"
        )

    except ImportError:
        raise ImportError(
            "xatlas is required for UV unwrapping but is not installed. "
            "Install with: pip install xatlas"
        )


def save_mesh_as_obj(mesh, output_path: str) -> str:
    """
    Save a trimesh mesh as .OBJ file (required format for Text2Tex/TEXTure).

    Args:
        mesh: trimesh.Trimesh object with UV coordinates.
        output_path: Directory to save the .OBJ file.

    Returns:
        Path to the saved .OBJ file.
    """
    import trimesh

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_path = output_dir / "mesh.obj"
    mesh.export(str(obj_path), file_type="obj")

    logger.info(f"Saved mesh as OBJ: {obj_path}")
    return str(obj_path)


def normalize_mesh(mesh) -> None:
    """
    Normalize mesh: center at origin and scale to unit bounding box.

    This is required by Text2Tex/TEXTure for consistent rendering.

    Args:
        mesh: trimesh.Trimesh object (modified in-place).
    """
    # Center at origin
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid

    # Scale to unit bounding box
    bounds = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    max_extent = bounds.max()
    if max_extent > 0:
        mesh.vertices /= max_extent

    logger.info(f"Normalized mesh to unit bounding box (max extent was {max_extent:.3f}).")
