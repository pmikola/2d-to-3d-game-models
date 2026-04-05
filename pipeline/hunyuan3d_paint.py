"""Hunyuan3D-2.1 Paint pipeline for PBR texture generation with MMGP offloading.

Wraps the official Hunyuan3D Paint code to generate albedo, normal, roughness,
and metallic textures for a given geometry mesh.  MMGP (Multi-Modal GPU
Partitioning) offloading is used to keep peak VRAM within 14 GB on systems with
32 GB of RAM.

Reference: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
VRAM: ~14 GB peak with MMGP ``LowRAM_LowVRAM`` on a 32 GB RAM system.
"""

import logging
import importlib
import os
import subprocess
import sys
from pathlib import Path
import types
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

REALESRGAN_X4PLUS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)


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

    @staticmethod
    def _prepare_paint_imports(code_root: Path) -> None:
        """
        Match Tencent's expected import layout for Paint.

        ``textureGenPipeline.py`` imports sibling modules like
        ``DifferentiableRenderer`` and ``utils`` as top-level packages, so
        ``hy3dpaint`` itself must be on ``sys.path``.  The custom rasterizer
        package also lives one level deeper.

        Critically, the ``utils`` package name is extremely common and can
        easily be shadowed by another ``utils`` module already in
        ``sys.modules`` (from pip-installed libraries, the CWD, etc.).  We
        therefore *eagerly* import the real ``hy3dpaint/utils`` package here
        and verify its ``__path__`` points to the correct directory so that
        all downstream ``from utils.<submod> import ...`` calls resolve to
        the Hunyuan Paint code.
        """
        paint_root = code_root / "hy3dpaint"
        rasterizer_root = paint_root / "custom_rasterizer"

        for path in (code_root, paint_root, rasterizer_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        # ----------------------------------------------------------------
        # Eagerly claim the ``utils`` top-level package for hy3dpaint/utils
        # ----------------------------------------------------------------
        expected_utils_dir = str(paint_root / "utils")
        utils_mod = sys.modules.get("utils")
        if utils_mod is not None:
            # Already in sys.modules — verify it points to the right place.
            existing_path = getattr(utils_mod, "__path__", None)
            if existing_path is None or expected_utils_dir not in [
                str(p) for p in existing_path
            ]:
                # Wrong ``utils`` (another library or a bare stub).  Replace
                # it with the real package from hy3dpaint/.
                logger.debug(
                    "Replacing incorrect sys.modules['utils'] (path=%s) "
                    "with hy3dpaint/utils (%s).",
                    existing_path,
                    expected_utils_dir,
                )
                del sys.modules["utils"]
                # Also remove any cached sub-modules that belonged to the
                # wrong parent.
                for key in list(sys.modules):
                    if key.startswith("utils."):
                        del sys.modules[key]
                utils_mod = None

        if utils_mod is None:
            # Import the real package.  Because ``paint_root`` is first on
            # ``sys.path``, ``import utils`` will find ``hy3dpaint/utils/``.
            try:
                import utils as _utils_pkg  # noqa: F401
            except ImportError:
                # If import fails (unlikely, __init__.py is trivial), create
                # a namespace-style stub with the correct __path__ so that
                # sub-module imports still work.
                _utils_pkg = types.ModuleType("utils")
                _utils_pkg.__path__ = [expected_utils_dir]
                _utils_pkg.__package__ = "utils"
                sys.modules["utils"] = _utils_pkg
                logger.debug(
                    "Created utils namespace stub with __path__=%s",
                    expected_utils_dir,
                )

        # Final safety check: guarantee __path__ is set correctly.
        utils_in_sys = sys.modules.get("utils")
        if utils_in_sys is not None and not getattr(utils_in_sys, "__path__", None):
            utils_in_sys.__path__ = [expected_utils_dir]
            utils_in_sys.__package__ = "utils"
            logger.debug(
                "Patched utils.__path__ to %s", expected_utils_dir
            )

        # ----------------------------------------------------------------
        # bpy stub — Hunyuan Paint imports ``bpy`` only for its optional
        # OBJ->GLB helper.  This project performs final GLB export itself,
        # so a lightweight stub keeps the import path working on systems
        # without Blender Python.
        # ----------------------------------------------------------------
        try:
            import bpy  # noqa: F401
        except ModuleNotFoundError:
            bpy_stub = types.ModuleType("bpy")
            # mesh_utils.py accesses bpy.data, bpy.context, bpy.ops, bpy.app
            # at function call time (not import time).  Provide nested stubs
            # so that the *import* of mesh_utils.py succeeds.  The functions
            # that use bpy will fail at call time, which is fine because this
            # project never calls convert_obj_to_glb through Blender.
            bpy_stub.data = types.ModuleType("bpy.data")
            bpy_stub.context = types.ModuleType("bpy.context")
            bpy_stub.ops = types.ModuleType("bpy.ops")
            bpy_stub.app = types.ModuleType("bpy.app")
            bpy_stub.app.version = (4, 0, 0)
            sys.modules.setdefault("bpy", bpy_stub)

        # ----------------------------------------------------------------
        # DifferentiableRenderer.mesh_inpaint_processor — the C++ UV inpaint
        # helper is an optional quality improvement.  If unavailable we keep
        # the pipeline usable by skipping the mesh-aware prefill step and
        # letting the later OpenCV inpaint handle holes.
        # ----------------------------------------------------------------
        try:
            importlib.import_module("DifferentiableRenderer.mesh_inpaint_processor")
        except (ModuleNotFoundError, ImportError):
            fallback_mod = types.ModuleType("DifferentiableRenderer.mesh_inpaint_processor")

            def meshVerticeInpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
                return texture, mask

            fallback_mod.meshVerticeInpaint = meshVerticeInpaint
            sys.modules["DifferentiableRenderer.mesh_inpaint_processor"] = fallback_mod

            # Also ensure the parent package is registered so that
            # ``from .mesh_inpaint_processor import ...`` inside
            # MeshRender.py can resolve correctly.
            if "DifferentiableRenderer" not in sys.modules:
                dr_pkg = types.ModuleType("DifferentiableRenderer")
                dr_pkg.__path__ = [str(paint_root / "DifferentiableRenderer")]
                dr_pkg.__package__ = "DifferentiableRenderer"
                sys.modules["DifferentiableRenderer"] = dr_pkg

            logger.warning(
                "mesh_inpaint_processor extension unavailable; using simplified UV "
                "inpaint fallback."
            )

    @staticmethod
    def _ensure_realesrgan_weights(code_root: Path) -> Path:
        """Download the RealESRGAN checkpoint on first use."""
        weight_path = code_root / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
        if weight_path.exists():
            return weight_path

        weight_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading RealESRGAN weights to %s", weight_path)
        urlretrieve(REALESRGAN_X4PLUS_URL, weight_path)
        logger.info("RealESRGAN weights downloaded successfully.")
        return weight_path

    @staticmethod
    def _summarize_output(output: str, max_lines: int = 20) -> str:
        lines = [line.rstrip() for line in output.splitlines() if line.strip()]
        return "\n".join(lines[-max_lines:])

    def _ensure_custom_rasterizer(self, code_root: Path) -> None:
        """
        Ensure Hunyuan Paint's CUDA rasterizer package is importable.

        Tries, in order:
        1. Import the pre-compiled ``custom_rasterizer`` package.
        2. Compile it from source (requires CUDA toolkit + C++ compiler).
        3. Register a pure-PyTorch fallback that is slower but requires no
           compilation.  The fallback implements the same ``rasterize_image``
           function as the C++ kernel and produces identical output.
        """
        try:
            import custom_rasterizer  # noqa: F401
            return
        except (ModuleNotFoundError, ImportError):
            pass

        # ----------------------------------------------------------
        # Attempt compilation
        # ----------------------------------------------------------
        compiled = False
        try:
            from torch.utils.cpp_extension import CUDA_HOME
        except ImportError:
            CUDA_HOME = None

        rasterizer_root = code_root / "hy3dpaint" / "custom_rasterizer"

        if CUDA_HOME is not None:
            logger.info("Installing Hunyuan Paint custom rasterizer from %s", rasterizer_root)
            cmd = [sys.executable, "-m", "pip", "install", "-e", str(rasterizer_root)]
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "CUDA_HOME": CUDA_HOME},
            )
            if result.returncode == 0:
                try:
                    import custom_rasterizer  # noqa: F401
                    compiled = True
                except (ModuleNotFoundError, ImportError):
                    pass
            if not compiled:
                logger.warning(
                    "Compilation of custom_rasterizer failed:\n%s",
                    self._summarize_output(result.stderr),
                )

        if compiled:
            return

        # ----------------------------------------------------------
        # Pure-PyTorch fallback
        # ----------------------------------------------------------
        logger.warning(
            "custom_rasterizer C++ extension is not available and cannot be "
            "compiled (CUDA_HOME=%s).  Using a pure-PyTorch software "
            "rasterizer fallback.  Texture baking will be significantly "
            "slower but functionally correct.",
            CUDA_HOME,
        )

        from .rasterizer_fallback import (
            rasterize_image,
            build_hierarchy,
            build_hierarchy_with_feat,
        )

        # Register the fallback as ``custom_rasterizer_kernel`` so that the
        # existing ``custom_rasterizer.render`` module can import it.
        kernel_mod = types.ModuleType("custom_rasterizer_kernel")
        kernel_mod.rasterize_image = rasterize_image
        kernel_mod.build_hierarchy = build_hierarchy
        kernel_mod.build_hierarchy_with_feat = build_hierarchy_with_feat
        sys.modules["custom_rasterizer_kernel"] = kernel_mod

        # Also register the ``custom_rasterizer`` package itself so that
        # ``import custom_rasterizer`` and ``custom_rasterizer.rasterize``
        # resolve correctly.
        cr_pkg = types.ModuleType("custom_rasterizer")
        cr_pkg.__path__ = [str(rasterizer_root / "custom_rasterizer")]
        sys.modules["custom_rasterizer"] = cr_pkg

        # Import the high-level render module that wraps the kernel.
        from custom_rasterizer.render import rasterize, interpolate  # noqa: F401
        cr_pkg.rasterize = rasterize
        cr_pkg.interpolate = interpolate

    @staticmethod
    def _install_realesrgan_fallback() -> None:
        """Register a pure-PIL fallback for ``image_super_utils.imageSuperNet``.

        RealESRGAN (and its dependency basicsr) often fails to install or
        import on Windows because basicsr requires C++ extensions that may
        not compile.  When that happens, we monkey-patch the upstream
        ``image_super_utils`` module so that the Paint pipeline still loads:
        instead of RealESRGAN 4x super-resolution, textures are upscaled
        with PIL Lanczos resampling.  Quality is slightly lower but the
        pipeline remains fully functional.

        NOTE: ``_prepare_paint_imports`` must be called first to ensure that
        ``sys.modules["utils"]`` already points to the real
        ``hy3dpaint/utils`` package with a correct ``__path__``.
        """
        # Check if realesrgan is actually importable.
        try:
            importlib.import_module("realesrgan")
            importlib.import_module("basicsr")
            return  # Both available — no fallback needed.
        except Exception:
            pass

        logger.warning(
            "realesrgan / basicsr are not importable (common on Windows). "
            "Texture super-resolution will use PIL Lanczos upscaling as a "
            "fallback.  To enable RealESRGAN, install manually:\n"
            "  pip install basicsr==1.4.2 realesrgan==0.3.0"
        )

        # Build a drop-in replacement module so that
        #   from utils.image_super_utils import imageSuperNet
        # inside the Paint code resolves without error.
        fallback_mod = types.ModuleType("utils.image_super_utils")

        class _LanczosSuperNet:
            """PIL-based 4x upscaler used when RealESRGAN is unavailable."""

            def __init__(self, config) -> None:
                self.scale = 4

            def __call__(self, image):
                w, h = image.size
                return image.resize(
                    (w * self.scale, h * self.scale), Image.LANCZOS
                )

        fallback_mod.imageSuperNet = _LanczosSuperNet
        sys.modules["utils.image_super_utils"] = fallback_mod

        # Attach the fallback as an attribute of the ``utils`` package.
        # ``_prepare_paint_imports`` already ensured that
        # ``sys.modules["utils"]`` is the real hy3dpaint/utils package
        # with the correct ``__path__``, so we just attach the sub-module.
        parent = sys.modules.get("utils")
        if parent is not None and not hasattr(parent, "image_super_utils"):
            parent.image_super_utils = fallback_mod

    def _ensure_python_dependencies(self, code_root: Path) -> None:
        """Install Python-side Paint dependencies that are safe to auto-setup."""
        from .deps import ensure_package

        ensure_package("pybind11", pip_spec="pybind11>=2.13.4")

        # RealESRGAN + basicsr: attempt to install, but tolerate failure
        # (common on Windows where basicsr's C++ extensions fail to compile).
        for pkg, spec in [("basicsr", "basicsr==1.4.2"), ("realesrgan", "realesrgan==0.3.0")]:
            try:
                ensure_package(pkg, pip_spec=spec)
            except (ImportError, RuntimeError) as exc:
                logger.warning(
                    "Optional dependency '%s' could not be installed: %s. "
                    "A PIL-based fallback will be used for texture upscaling.",
                    pkg,
                    exc,
                )

        # If realesrgan still cannot be imported, register the PIL fallback
        # *before* the Paint pipeline tries to import image_super_utils.
        self._install_realesrgan_fallback()

        # Only download the ~67 MB RealESRGAN checkpoint if the package is
        # actually usable; the PIL fallback does not need it.
        try:
            importlib.import_module("realesrgan")
            self._ensure_realesrgan_weights(code_root)
        except ImportError:
            pass

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

        code_root_path = Path(code_root)
        self._prepare_paint_imports(code_root_path)
        self._ensure_python_dependencies(code_root_path)

        try:
            from hy3dpaint.textureGenPipeline import (
                Hunyuan3DPaintConfig,
                Hunyuan3DPaintPipeline,
            )
        except ModuleNotFoundError as exc:
            missing_name = exc.name or "an internal Hunyuan3D Paint dependency"
            raise ImportError(
                "Hunyuan3D Paint dependencies are not fully available. "
                f"Missing module: {missing_name}\n"
                "Paint expects the local repo checkout plus compiled renderer "
                "extensions. Ensure these steps have been completed:\n"
                "  1. cd Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && pip install -e .\n"
                "  2. cd Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh\n"
                "  3. Download RealESRGAN_x4plus.pth into hy3dpaint/ckpt/\n"
                "After that, rerun the full backend."
            ) from exc

        self._ensure_custom_rasterizer(code_root_path)

        logger.info("Configuring Hunyuan3D Paint pipeline...")
        config = Hunyuan3DPaintConfig(max_num_view=self.max_views, resolution=self.resolution)
        config.device = "cuda" if self.device_config.has_gpu else "cpu"
        config.multiview_pretrained_path = self.model_path

        # Adjust paths relative to code root
        config.multiview_cfg_path = str(
            code_root_path / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"
        )
        config.custom_pipeline = str(
            code_root_path / "hy3dpaint" / "hunyuanpaintpbr"
        )
        config.realesrgan_ckpt_path = str(
            code_root_path / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
        )

        logger.info("Loading Hunyuan3D Paint pipeline...")
        self.pipeline = Hunyuan3DPaintPipeline(config)

        # Auto-install and apply MMGP offloading to keep VRAM under budget
        from .deps import ensure_package

        try:
            ensure_package("mmgp", pip_spec="mmgp>=0.9.0")
            from mmgp import offload, profile_type

            profile = getattr(profile_type, self.mmgp_profile, profile_type.LowRAM_LowVRAM)
            offload.profile(self.pipeline, profile)
            logger.info(f"MMGP offloading enabled ({self.mmgp_profile}).")
        except Exception as exc:
            logger.warning(
                "MMGP offloading unavailable (%s). Paint pipeline runs without "
                "memory offloading — may OOM on GPUs < 24GB.",
                exc,
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
                    save_glb=False,
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
