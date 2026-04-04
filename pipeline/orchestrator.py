"""
Main pipeline orchestrator.

Chains the full 2D-to-3D pipeline:
    1. Preprocess: background removal, resize, quality check
    2. Geometry: Hi3DGen — image to 3D mesh
    3. UV Unwrap: xatlas UV mapping for texturing
    4. Texturing: Text2Tex / TEXTure — diffusion-based multi-view texturing
    5. Export: GLB with embedded textures for Blender
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .device import DeviceConfig, detect_device, log_device_info
from .export import export_textured_dir_to_glb, export_to_glb, validate_glb
from .geometry import Hi3DGenWrapper, normalize_mesh, save_mesh_as_obj, unwrap_uvs
from .mesh_repair import repair_and_prepare
from .pbr_maps import generate_pbr_maps, save_pbr_maps
from .preprocess import preprocess_image
from .texturing import TextureGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    # Pipeline backend: "hi3dgen" (default) or "hunyuan3d"
    backend: str = "hi3dgen"

    # Preprocessing
    target_size: int = 512
    remove_background: bool = True

    # Geometry
    geometry_seed: int = 42
    geometry_guidance_scale: float = 7.5
    geometry_steps: int = 50

    # Texturing
    texture_prompt: str | None = None  # Auto-generated if None
    texture_seed: int = 42

    # Mesh repair
    mesh_repair: bool = True
    mesh_smooth_iterations: int = 3
    mesh_decimate_ratio: float | None = None  # None = no decimation

    # PBR map generation
    generate_pbr: bool = True

    # Paths
    hi3dgen_path: str | None = None
    text2tex_path: str | None = None
    hunyuan3d_model_path: str | None = None

    # Runtime
    force_cpu: bool = False
    skip_texturing: bool = False


@dataclass
class PipelineResult:
    """Result from processing a single image."""

    input_path: str
    output_path: str | None = None
    success: bool = False
    error: str | None = None
    duration_seconds: float = 0.0
    mesh_vertices: int = 0
    mesh_faces: int = 0
    glb_size_mb: float = 0.0


class Pipeline:
    """
    Main 2D-to-3D pipeline orchestrator.

    Usage:
        pipeline = Pipeline(config)
        result = pipeline.process_image("photo.png", "output/model.glb")

        # Or batch:
        results = pipeline.process_batch("./images/", "./output/")
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.device_config: DeviceConfig | None = None
        self.geometry_generator: Hi3DGenWrapper | None = None
        self.texture_generator: TextureGenerator | None = None
        self._hunyuan3d = None  # Lazy-loaded Hunyuan3D wrapper

    def initialize(self) -> None:
        """Initialize device detection and model loading."""
        logger.info("=" * 60)
        logger.info("Initializing 2D-to-3D Pipeline")
        logger.info(f"  Backend: {self.config.backend}")
        logger.info("=" * 60)

        # Detect hardware
        self.device_config = detect_device(force_cpu=self.config.force_cpu)
        log_device_info(self.device_config)

        if self.config.backend == "hunyuan3d":
            from .hunyuan3d import Hunyuan3DWrapper

            self._hunyuan3d = Hunyuan3DWrapper(
                device_config=self.device_config,
                model_path=self.config.hunyuan3d_model_path,
            )
            logger.info("Using Hunyuan3D-2.1 backend (geometry + PBR texturing).")
        else:
            # Default: Hi3DGen + Text2Tex two-stage pipeline
            self.geometry_generator = Hi3DGenWrapper(
                device_config=self.device_config,
                hi3dgen_path=self.config.hi3dgen_path,
            )
            if not self.config.skip_texturing:
                self.texture_generator = TextureGenerator(
                    device_config=self.device_config,
                    text2tex_path=self.config.text2tex_path,
                )

        logger.info("Pipeline initialized.")

    def process_image(self, input_path: str, output_path: str) -> PipelineResult:
        """
        Process a single image through the full pipeline.

        Args:
            input_path: Path to the input image (PNG/JPG/WEBP).
            output_path: Path for the output .GLB file.

        Returns:
            PipelineResult with status and metadata.
        """
        start_time = time.time()
        result = PipelineResult(input_path=input_path)

        try:
            if self.device_config is None:
                self.initialize()

            logger.info("-" * 40)
            logger.info(f"Processing: {input_path}")
            logger.info("-" * 40)

            # Stage 0: Preprocess
            logger.info("[Stage 0/4] Preprocessing image...")
            preprocessed = preprocess_image(
                image_path=input_path,
                target_size=self.config.target_size,
                remove_bg=self.config.remove_background,
                use_gpu=self.device_config.has_gpu,
            )

            # Save preprocessed image for debugging
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            preprocessed_path = output_dir / f"{Path(input_path).stem}_preprocessed.png"
            preprocessed.save(str(preprocessed_path))
            logger.info(f"Saved preprocessed image: {preprocessed_path}")

            # --- Hunyuan3D backend (single-stage geometry + PBR texturing) ---
            if self.config.backend == "hunyuan3d" and self._hunyuan3d is not None:
                logger.info("[Stage 1/2] Generating geometry + textures (Hunyuan3D-2.1)...")
                hy_result = self._hunyuan3d.generate(
                    image=preprocessed,
                    seed=self.config.geometry_seed,
                    guidance_scale=self.config.geometry_guidance_scale,
                    num_steps=self.config.geometry_steps,
                )
                mesh = hy_result["mesh"]
                result.mesh_vertices = len(mesh.vertices)
                result.mesh_faces = len(mesh.faces)

                # Mesh repair
                if self.config.mesh_repair:
                    logger.info("[Stage 1.5/2] Repairing mesh...")
                    mesh = repair_and_prepare(
                        mesh,
                        decimate_ratio=self.config.mesh_decimate_ratio,
                        smooth_iterations=self.config.mesh_smooth_iterations,
                    )

                normalize_mesh(mesh)

                # Export with Hunyuan3D's PBR textures if available
                with tempfile.TemporaryDirectory() as tmp_dir:
                    obj_path = save_mesh_as_obj(mesh, tmp_dir)
                    texture_path = None
                    if hy_result.get("texture"):
                        texture_path = str(Path(tmp_dir) / "albedo.png")
                        hy_result["texture"].save(texture_path)

                    # Save PBR maps from Hunyuan3D output
                    pbr_dir = Path(tmp_dir) / "pbr"
                    pbr_dir.mkdir(exist_ok=True)
                    for map_name in ("normal_map", "metallic_map", "roughness_map"):
                        if hy_result.get(map_name):
                            hy_result[map_name].save(str(pbr_dir / f"{map_name}.png"))

                    logger.info("[Stage 2/2] Exporting to GLB...")
                    export_to_glb(obj_path, texture_path, output_path)

            # --- Hi3DGen + Text2Tex backend (two-stage) ---
            else:
                logger.info("[Stage 1/5] Generating 3D geometry (Hi3DGen)...")
                mesh = self.geometry_generator.generate_mesh(
                    image=preprocessed,
                    seed=self.config.geometry_seed,
                    guidance_scale=self.config.geometry_guidance_scale,
                    num_inference_steps=self.config.geometry_steps,
                )

                result.mesh_vertices = len(mesh.vertices)
                result.mesh_faces = len(mesh.faces)

                # Mesh repair (new)
                if self.config.mesh_repair:
                    logger.info("[Stage 2/5] Repairing mesh...")
                    mesh = repair_and_prepare(
                        mesh,
                        decimate_ratio=self.config.mesh_decimate_ratio,
                        smooth_iterations=self.config.mesh_smooth_iterations,
                    )
                    result.mesh_vertices = len(mesh.vertices)
                    result.mesh_faces = len(mesh.faces)

                # Normalize mesh
                normalize_mesh(mesh)

                # UV unwrap for texturing
                logger.info("[Stage 3/5] UV unwrapping...")
                unwrap_uvs(mesh)

                # Save intermediate OBJ
                with tempfile.TemporaryDirectory() as tmp_dir:
                    obj_path = save_mesh_as_obj(mesh, tmp_dir)

                    # Texturing
                    if not self.config.skip_texturing and self.texture_generator is not None:
                        logger.info("[Stage 4/5] Generating textures (Text2Tex/diffusers)...")
                        textured_dir = self.texture_generator.generate_texture(
                            mesh_obj_path=obj_path,
                            output_dir=str(Path(tmp_dir) / "textured"),
                            prompt=self.config.texture_prompt,
                            original_image=preprocessed,
                            seed=self.config.texture_seed,
                        )

                        # PBR map generation (new)
                        if self.config.generate_pbr:
                            texture_atlas = Path(textured_dir) / "texture_atlas.png"
                            if texture_atlas.exists():
                                logger.info("[Stage 4.5/5] Generating PBR maps...")
                                from PIL import Image as PILImage
                                albedo = PILImage.open(texture_atlas)
                                pbr_maps = generate_pbr_maps(albedo)
                                save_pbr_maps(pbr_maps, str(Path(textured_dir) / "pbr"))

                        # Export to GLB
                        logger.info("[Stage 5/5] Exporting to GLB...")
                        export_textured_dir_to_glb(textured_dir, output_path)
                    else:
                        # Export geometry-only GLB
                        logger.info("[Stage 5/5] Exporting geometry-only GLB...")
                        export_to_glb(obj_path, None, output_path)

            # Validate output
            validation = validate_glb(output_path)
            if validation.get("valid"):
                result.glb_size_mb = validation.get("file_size_mb", 0.0)

            result.output_path = output_path
            result.success = True

            duration = time.time() - start_time
            result.duration_seconds = duration

            logger.info(f"SUCCESS: {input_path} -> {output_path}")
            logger.info(
                f"  Vertices: {result.mesh_vertices}, Faces: {result.mesh_faces}, "
                f"Size: {result.glb_size_mb:.1f} MB, Time: {duration:.1f}s"
            )

        except Exception as e:
            result.error = str(e)
            result.duration_seconds = time.time() - start_time
            logger.error(f"FAILED: {input_path} — {e}")

        return result

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*",
    ) -> list[PipelineResult]:
        """
        Process a batch of images from a directory.

        Args:
            input_dir: Directory containing input images.
            output_dir: Directory for output .GLB files.
            file_pattern: Glob pattern for filtering files (default: all supported formats).

        Returns:
            List of PipelineResult for each processed image.
        """
        from .preprocess import SUPPORTED_FORMATS

        input_path = Path(input_dir)
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        # Find all supported images
        images = []
        for fmt in SUPPORTED_FORMATS:
            images.extend(input_path.glob(f"*{fmt}"))
            images.extend(input_path.glob(f"*{fmt.upper()}"))
        images = sorted(set(images))

        if not images:
            logger.warning(f"No supported images found in {input_dir}")
            return []

        logger.info(f"Found {len(images)} images to process in {input_dir}")

        # Initialize once for the batch
        if self.device_config is None:
            self.initialize()

        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(images):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Image {i + 1}/{len(images)}: {img_path.name}")
            logger.info(f"{'=' * 60}")

            glb_output = output_path / f"{img_path.stem}.glb"
            result = self.process_image(str(img_path), str(glb_output))
            results.append(result)

            if not result.success:
                logger.warning(f"Skipping failed image: {img_path.name} — {result.error}")

        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = sum(r.duration_seconds for r in results)

        logger.info(f"\n{'=' * 60}")
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"  Total: {len(results)} | Success: {successful} | Failed: {failed}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"{'=' * 60}")

        return results
