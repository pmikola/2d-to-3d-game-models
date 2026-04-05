"""
Main pipeline orchestrator.

Chains the full 2D-to-3D pipeline:
    1. Preprocess: background removal, resize, quality check
    2. Geometry: Hunyuan3D-2.1 fp16 shape generation
    3. Optional Hunyuan game-ready decimation + normalization
    4. Export: geometry-only GLB for Blender

Legacy Hi3DGen + Text2Tex stages remain available as optional backends.
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .device import DeviceConfig, detect_device, log_device_info, release_runtime_memory
from .export import (
    export_hunyuan_paint_to_glb,
    export_textured_dir_to_glb,
    export_to_glb,
    validate_glb,
)
from .geometry import Hi3DGenWrapper, normalize_mesh, save_mesh_as_obj, unwrap_uvs
from .mesh_repair import (
    decimate_mesh,
    make_watertight,
    prepare_game_ready_mesh,
    remove_small_components,
    repair_and_prepare,
)
from .pbr_maps import generate_pbr_maps, save_pbr_maps
from .preprocess import composite_on_background, preprocess_image, remove_gray_background
from .texturing import TextureGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    # Pipeline backend: "hunyuan3d" (default) or "hi3dgen"
    backend: str = "hunyuan3d"

    # Preprocessing
    target_size: int = 512
    remove_background: bool = True
    bg_model: str = "birefnet-general"  # rembg session name for background removal

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
    generate_pbr: bool = False

    # MV-Adapter multi-view generation (full backend)
    # 30 steps with ShiftSNR scheduler provides ~95% of the quality of 50
    # steps while cutting MV-Adapter runtime by ~40% (~18-36s savings).
    zero123_steps: int = 30
    zero123_guidance_scale: float = 3.0

    # Dynamic range preprocessing
    correct_exposure: bool = False
    exposure_low_percentile: float = 1.0
    exposure_high_percentile: float = 99.0

    # Hunyuan3D-2mv shape
    shape_octree_resolution: int = 384

    # Hunyuan3D Paint texturing
    paint_max_views: int = 6
    paint_resolution: int = 512
    paint_use_remesh: bool = True
    mmgp_profile: str = "LowRAM_LowVRAM"

    # Paths
    hi3dgen_path: str | None = None
    text2tex_path: str | None = None
    hunyuan3d_model_path: str | None = None  # Local path or Hugging Face repo ID

    # Runtime
    force_cpu: bool = False
    skip_texturing: bool = True
    game_ready: bool = True
    game_ready_target_faces: int = 50000


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
        self._zero123 = None  # MV-Adapter multi-view (full backend)
        self._hunyuan3d_paint = None  # Hunyuan3D Paint texturing (full backend)

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

            if not self.config.skip_texturing:
                logger.info(
                    "Hunyuan3D-2.1 is configured as a shape-only backend in this "
                    "project. Disabling texture generation."
                )
                self.config.skip_texturing = True

            if self.config.generate_pbr:
                logger.info(
                    "Disabling PBR map generation for the Hunyuan3D-2.1 shape-only "
                    "profile."
                )
                self.config.generate_pbr = False

            if self.config.game_ready:
                if self.config.game_ready_target_faces < 4:
                    raise ValueError("game_ready_target_faces must be >= 4.")
                logger.info(
                    "Game-ready export enabled for Hunyuan3D-2.1 "
                    "(target faces: %d).",
                    self.config.game_ready_target_faces,
                )

            if self.config.mesh_repair:
                logger.info(
                    "Disabling mesh repair for the Hunyuan3D-2.1 backend. "
                    "The legacy repair pass can over-simplify Hunyuan meshes."
                )
                self.config.mesh_repair = False

            self._hunyuan3d = Hunyuan3DWrapper(
                device_config=self.device_config,
                model_path=self.config.hunyuan3d_model_path,
            )
            logger.info(
                "Using Hunyuan3D-2.1 backend (fp16 shape generation, geometry-only export)."
            )
        elif self.config.backend == "full":
            from .hunyuan3d import Hunyuan3DWrapper
            from .hunyuan3d_paint import Hunyuan3DPaintWrapper
            from .zero123plus import MVAdapterWrapper

            self._zero123 = MVAdapterWrapper(
                device_config=self.device_config,
            )
            self._hunyuan3d = Hunyuan3DWrapper(
                device_config=self.device_config,
                model_path=self.config.hunyuan3d_model_path,
                variant="multiview",
            )
            self._hunyuan3d_paint = Hunyuan3DPaintWrapper(
                device_config=self.device_config,
                model_path=self.config.hunyuan3d_model_path,
                max_views=self.config.paint_max_views,
                resolution=self.config.paint_resolution,
                mmgp_profile=self.config.mmgp_profile,
            )

            # Full pipeline does texturing + PBR
            self.config.skip_texturing = False
            self.config.generate_pbr = True
            self.texture_generator = TextureGenerator(
                device_config=self.device_config,
                text2tex_path=self.config.text2tex_path,
            )

            # NOTE: game_ready decimation is no longer disabled for the
            # full backend.  Stage 3 always decimates to
            # game_ready_target_faces (default 50K) to keep intermediate
            # files small and ensure the fallback texture path is fast.
            # Paint's own remesher handles pre-decimated input fine.
            if self.config.game_ready:
                logger.info(
                    "Full backend: game-ready decimation enabled "
                    "(target faces: %d).  Paint will re-remesh internally.",
                    self.config.game_ready_target_faces,
                )

            if self.config.mesh_repair:
                logger.info(
                    "Disabling legacy mesh repair for the full backend to "
                    "preserve Hunyuan3D-2mv geometry fidelity before UV/texturing."
                )
                self.config.mesh_repair = False

            if (
                self.device_config.has_gpu
                and self.device_config.vram_gb < 18
                and self.config.zero123_steps > 35
            ):
                logger.info(
                    "Reducing MV-Adapter steps from %d to 35 for stability on %.1f GB VRAM.",
                    self.config.zero123_steps,
                    self.device_config.vram_gb,
                )
                self.config.zero123_steps = 35

            logger.info(
                "Full 5-stage pipeline selected: MV-Adapter -> Hunyuan3D-2mv -> "
                "mesh repair -> Hunyuan3D Paint -> PBR GLB export."
            )

        elif self.config.backend == "triposg":
            from .triposg import TripoSGWrapper

            self.geometry_generator = TripoSGWrapper(
                device_config=self.device_config,
            )
            logger.info("Using TripoSG backend (geometry only, ~8GB VRAM).")
        else:
            if self.config.game_ready:
                logger.info(
                    "Game-ready decimation currently only applies to the "
                    "Hunyuan3D backend; ignoring it for %s.",
                    self.config.backend,
                )
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

            # Stage 0: Preprocess — returns RGBA with alpha preserved
            logger.info("[Stage 0] Preprocessing image...")
            preprocessed_rgba = preprocess_image(
                image_path=input_path,
                target_size=self.config.target_size,
                remove_bg=self.config.remove_background,
                use_gpu=self.device_config.has_gpu,
                correct_exposure=self.config.correct_exposure,
                exposure_low_percentile=self.config.exposure_low_percentile,
                exposure_high_percentile=self.config.exposure_high_percentile,
                bg_model=self.config.bg_model,
            )

            # Save debug image (composite on white for visibility)
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            preprocessed_path = output_dir / f"{Path(input_path).stem}_preprocessed.png"
            debug_img = composite_on_background(preprocessed_rgba, (255, 255, 255))
            debug_img.save(str(preprocessed_path))
            logger.info(f"Saved preprocessed image: {preprocessed_path}")

            # --- Hunyuan3D backend (shape generation only) ---
            if self.config.backend == "hunyuan3d" and self._hunyuan3d is not None:
                # Hunyuan3D shape-only expects RGB on white background
                preprocessed = composite_on_background(preprocessed_rgba, (255, 255, 255))
                logger.info("[Stage 1/2] Generating 3D geometry (Hunyuan3D-2.1 fp16)...")
                try:
                    self._hunyuan3d.load()
                    hy_result = self._hunyuan3d.generate(
                        image=preprocessed,
                        seed=self.config.geometry_seed,
                        guidance_scale=self.config.geometry_guidance_scale,
                        num_steps=self.config.geometry_steps,
                    )
                finally:
                    self._hunyuan3d.unload()
                    release_runtime_memory("after_hunyuan_shape_stage")
                mesh = hy_result["mesh"]
                result.mesh_vertices = len(mesh.vertices)
                result.mesh_faces = len(mesh.faces)

                if self.config.game_ready:
                    logger.info("[Stage 1.5/2] Optimizing mesh for game-ready export...")
                    mesh = prepare_game_ready_mesh(
                        mesh,
                        target_face_count=self.config.game_ready_target_faces,
                    )
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
                    result.mesh_vertices = len(mesh.vertices)
                    result.mesh_faces = len(mesh.faces)

                normalize_mesh(mesh)
                result.mesh_vertices = len(mesh.vertices)
                result.mesh_faces = len(mesh.faces)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    obj_path = save_mesh_as_obj(mesh, tmp_dir)
                    logger.info("[Stage 2/2] Exporting geometry-only GLB...")
                    export_to_glb(obj_path, None, output_path)

            # --- Full 5-stage backend ---
            elif self.config.backend == "full" and self._zero123 is not None:
                # Stage 1: Multi-view generation (MV-Adapter ~14GB)
                # MV-Adapter wants RGBA (it composites on gray internally)
                logger.info("[Stage 1/5] Generating multi-view images (MV-Adapter)...")
                mv_view_names = ["front", "left", "back", "right"]
                mv_batch_size = 2 if self.device_config.has_gpu and self.device_config.vram_gb < 18 else None
                try:
                    views = self._zero123.generate_views(
                        preprocessed_rgba,
                        num_inference_steps=self.config.zero123_steps,
                        guidance_scale=self.config.zero123_guidance_scale,
                        requested_view_names=mv_view_names,
                        batch_size=mv_batch_size,
                    )
                finally:
                    self._zero123.unload()
                    release_runtime_memory("after_mvadapter_stage")

                # Save multi-view debug images — include azimuth in filename so
                # it is easy to diagnose which azimuth produces which view.
                _name_to_az = {
                    name: az for _idx, (name, az) in self._zero123.AZIMUTH_MAP.items()
                }
                for view_name, view_img in views.items():
                    az_deg = _name_to_az.get(view_name, "?")
                    view_path = output_dir / f"{Path(input_path).stem}_view_{view_name}_az{az_deg}.png"
                    view_img.save(str(view_path))
                logger.info(f"Saved {len(views)} multi-view images for debugging.")

                # Remove gray background from MV-Adapter views before Hunyuan3D-2mv.
                # MV-Adapter outputs RGB on gray (128) bg; Hunyuan3D-2mv's
                # MVImageProcessorV2 expects RGBA with meaningful alpha.
                cardinal_views = {}
                for name in ("front", "left", "back", "right"):
                    if name in views:
                        cardinal_views[name] = remove_gray_background(views[name])
                del views
                release_runtime_memory("after_multiview_output_cleanup")

                # Stage 2: Shape generation (Hunyuan3D-2mv ~10GB)
                logger.info("[Stage 2/5] Generating 3D shape (Hunyuan3D-2mv)...")
                try:
                    self._hunyuan3d.load()
                    hy_result = self._hunyuan3d.generate_from_multiview(
                        views=cardinal_views,
                        seed=self.config.geometry_seed,
                        guidance_scale=self.config.geometry_guidance_scale,
                        num_steps=self.config.geometry_steps,
                        octree_resolution=self.config.shape_octree_resolution,
                    )
                    mesh = hy_result["mesh"]
                    result.mesh_vertices = len(mesh.vertices)
                    result.mesh_faces = len(mesh.faces)
                finally:
                    self._hunyuan3d.unload()
                    release_runtime_memory("after_hunyuan_multiview_stage")
                del hy_result
                del cardinal_views
                release_runtime_memory("after_shape_output_cleanup")

                # Stage 3: Mesh prep — normalize and ALWAYS decimate.
                #
                # We always decimate to game_ready_target_faces (default 50K)
                # even though Hunyuan3D Paint does its own remeshing, because:
                #   1. Smaller intermediate GLB = faster I/O
                #   2. If Paint fails, the fallback path gets a manageable mesh
                #      instead of 400K+ faces (which kills xatlas UV unwrap)
                #   3. Paint's remesher handles pre-decimated input fine
                #
                # UV unwrapping is still skipped when Paint will redo it
                # (paint_use_remesh=True).  The fallback path handles UV
                # unwrapping on the already-decimated mesh if Paint fails.
                stage3_t0 = time.time()
                logger.info("[Stage 3/5] Preparing mesh for texturing...")

                # Always decimate to a reasonable face count for the full
                # backend.  This is critical: raw Hunyuan3D-2mv meshes can
                # have 400K+ faces which causes extreme slowness in every
                # downstream operation (UV unwrap, texture baking, GLB I/O).
                decimate_target = self.config.game_ready_target_faces
                pre_decimate_faces = len(mesh.faces)
                if pre_decimate_faces > decimate_target:
                    decimate_t0 = time.time()
                    logger.info(
                        "[Stage 3/5] Decimating mesh: %d -> %d target faces...",
                        pre_decimate_faces,
                        decimate_target,
                    )
                    mesh = prepare_game_ready_mesh(
                        mesh,
                        target_face_count=decimate_target,
                    )
                    decimate_elapsed = time.time() - decimate_t0
                    logger.info(
                        "[Stage 3/5] Decimation complete in %.1fs: %d -> %d faces.",
                        decimate_elapsed,
                        pre_decimate_faces,
                        len(mesh.faces),
                    )
                else:
                    logger.info(
                        "[Stage 3/5] Mesh already at %d faces (<= %d target); "
                        "skipping decimation.",
                        pre_decimate_faces,
                        decimate_target,
                    )

                # Lightweight mesh repair: remove disconnected components,
                # merge close vertices, and fix normals.  These are safe
                # operations that won't damage geometry but fix common
                # Hunyuan3D-2mv output issues (multiple components, gaps).
                repair_t0 = time.time()
                pre_repair_faces = len(mesh.faces)
                pre_repair_verts = len(mesh.vertices)

                mesh = remove_small_components(mesh, min_face_ratio=0.05)
                mesh = make_watertight(mesh)  # merge close vertices + fill holes

                try:
                    import trimesh
                    trimesh.repair.fix_normals(mesh)
                except Exception:
                    pass

                repair_elapsed = time.time() - repair_t0
                logger.info(
                    "[Stage 3/5] Lightweight repair complete in %.1fs: "
                    "%d -> %d verts, %d -> %d faces.",
                    repair_elapsed,
                    pre_repair_verts,
                    len(mesh.vertices),
                    pre_repair_faces,
                    len(mesh.faces),
                )

                normalize_t0 = time.time()
                normalize_mesh(mesh)
                logger.info(
                    "[Stage 3/5] Normalization complete in %.1fs.",
                    time.time() - normalize_t0,
                )

                # Only UV-unwrap here when Paint will NOT redo it.
                skip_uv = self.config.paint_use_remesh
                if skip_uv:
                    logger.info(
                        "[Stage 3/5] Skipping UV unwrap — Hunyuan3D Paint will "
                        "remesh and UV-unwrap internally (paint_use_remesh=True)."
                    )
                else:
                    uv_t0 = time.time()
                    unwrap_uvs(mesh)
                    logger.info(
                        "[Stage 3/5] UV unwrap complete in %.1fs.",
                        time.time() - uv_t0,
                    )

                result.mesh_vertices = len(mesh.vertices)
                result.mesh_faces = len(mesh.faces)
                stage3_elapsed = time.time() - stage3_t0
                logger.info(
                    "[Stage 3/5] Mesh preparation complete in %.1fs "
                    "(%d verts, %d faces, UV unwrap %s).",
                    stage3_elapsed,
                    result.mesh_vertices,
                    result.mesh_faces,
                    "skipped" if skip_uv else "done",
                )

                # Stage 4: PBR texturing (Hunyuan3D Paint ~14GB with MMGP)
                logger.info("[Stage 4/5] Generating PBR textures (Hunyuan3D Paint)...")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    intermediate_obj = save_mesh_as_obj(mesh, tmp_dir)
                    texture_dir = str(Path(tmp_dir) / "textured")
                    paint_result = None
                    paint_error = None

                    # Paint accepts OBJ directly — skip the unnecessary
                    # intermediate GLB export (~1-3s saved).
                    paint_mesh_path = intermediate_obj

                    # Paint wants RGB on white (it has its own delighting)
                    paint_reference = composite_on_background(preprocessed_rgba, (255, 255, 255))

                    try:
                        self._hunyuan3d_paint.load()
                        paint_result = self._hunyuan3d_paint.generate_textures(
                            mesh_path=paint_mesh_path,
                            reference_image=paint_reference,
                            output_dir=texture_dir,
                            use_remesh=self.config.paint_use_remesh,
                        )
                    except Exception as exc:
                        paint_error = exc
                        logger.warning(
                            "Hunyuan3D Paint unavailable (%s). Falling back to the "
                            "built-in depth-conditioned texture generator.",
                            exc,
                        )
                    finally:
                        self._hunyuan3d_paint.unload()
                        release_runtime_memory("after_paint_stage")

                    if paint_result is not None:
                        # Stage 5: Final PBR GLB export (CPU)
                        logger.info("[Stage 5/5] Exporting final PBR GLB...")
                        export_hunyuan_paint_to_glb(
                            texture_dir, output_path, paint_result
                        )
                    else:
                        # Paint failed — the fallback texture generator needs
                        # UV-mapped geometry.  Stage 3 already decimated the
                        # mesh, but as a safety net we verify the face count
                        # is manageable before running the expensive UV unwrap.
                        fallback_t0 = time.time()
                        logger.info(
                            "Paint failed; preparing mesh for fallback texture "
                            "generator (%d faces)...",
                            len(mesh.faces),
                        )

                        # Safety decimation: ensure face count is reasonable
                        # for xatlas UV unwrap (target 50K, tolerate up to
                        # 80K before forcing another decimation pass).
                        fallback_max_faces = int(
                            self.config.game_ready_target_faces * 1.6
                        )
                        if len(mesh.faces) > fallback_max_faces:
                            logger.info(
                                "Fallback: mesh has %d faces (> %d limit); "
                                "decimating to %d...",
                                len(mesh.faces),
                                fallback_max_faces,
                                self.config.game_ready_target_faces,
                            )
                            fb_dec_t0 = time.time()
                            mesh = decimate_mesh(
                                mesh,
                                target_face_count=self.config.game_ready_target_faces,
                            )
                            logger.info(
                                "Fallback: decimation done in %.1fs (%d faces).",
                                time.time() - fb_dec_t0,
                                len(mesh.faces),
                            )

                        # Now UV-unwrap if we skipped it in Stage 3.
                        if skip_uv:
                            fb_uv_t0 = time.time()
                            logger.info(
                                "Fallback: running deferred UV unwrap on %d faces...",
                                len(mesh.faces),
                            )
                            unwrap_uvs(mesh)
                            logger.info(
                                "Fallback: UV unwrap complete in %.1fs.",
                                time.time() - fb_uv_t0,
                            )

                        # Always re-save the OBJ after fallback decimation
                        # and/or UV unwrap so the texture generator gets the
                        # current mesh (not the pre-decimation version from
                        # Stage 3).
                        intermediate_obj = save_mesh_as_obj(mesh, tmp_dir)

                        # Update result counts to reflect fallback mesh state.
                        result.mesh_vertices = len(mesh.vertices)
                        result.mesh_faces = len(mesh.faces)

                        # Reduce viewpoints for faster fallback texturing.
                        # The full pipeline uses up to 36 viewpoints which is
                        # extremely slow; 8 viewpoints covers all cardinal
                        # directions and is much faster for a fallback path.
                        original_viewpoints = self.device_config.texture_num_viewpoints
                        fallback_viewpoints = min(8, original_viewpoints)
                        if original_viewpoints != fallback_viewpoints:
                            logger.info(
                                "Fallback: reducing viewpoints from %d to %d "
                                "for faster texture generation.",
                                original_viewpoints,
                                fallback_viewpoints,
                            )
                            self.device_config.texture_num_viewpoints = fallback_viewpoints

                        if self.texture_generator is None:
                            self.texture_generator = TextureGenerator(
                                device_config=self.device_config,
                                text2tex_path=self.config.text2tex_path,
                            )
                        try:
                            fallback_dir = self.texture_generator.generate_texture(
                                mesh_obj_path=intermediate_obj,
                                output_dir=str(Path(tmp_dir) / "textured_fallback"),
                                prompt=self.config.texture_prompt,
                                original_image=paint_reference,
                                seed=self.config.texture_seed,
                            )
                        finally:
                            # Restore original viewpoint count so it doesn't
                            # affect any subsequent runs in a batch.
                            self.device_config.texture_num_viewpoints = original_viewpoints
                            # Release the SD2 pipeline VRAM loaded by the
                            # fallback TextureGenerator.
                            release_runtime_memory("after_fallback_texture_stage")

                        fallback_elapsed = time.time() - fallback_t0
                        logger.info(
                            "[Stage 5/5] Exporting textured GLB via fallback "
                            "texture pipeline (fallback took %.1fs)...",
                            fallback_elapsed,
                        )
                        export_textured_dir_to_glb(fallback_dir, output_path)
                    del paint_result
                    del paint_error
                    del intermediate_obj
                    release_runtime_memory("after_texturing_output_cleanup")

            # --- Hi3DGen + Text2Tex backend (two-stage) ---
            else:
                # Hi3DGen / TripoSG backends expect RGB on white background
                preprocessed = composite_on_background(preprocessed_rgba, (255, 255, 255))
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
