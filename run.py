#!/usr/bin/env python3
"""
2D-to-3D Game Models Pipeline — CLI Entry Point

Convert 2D images (PNG/JPG) to 3D shape models (.GLB) for Blender.

Default pipeline:
  Stage 1: Hunyuan3D-2.1 fp16 — high-quality 3D shape generation
  Stage 2: Game-ready decimation + geometry-only GLB export

Legacy pipeline:
  Hi3DGen + Text2Tex remains available via `--backend hi3dgen`

Usage:
  python run.py --input photo.png --output model.glb
  python run.py --batch-dir ./images/ --output-dir ./models/
  python run.py --input photo.png --output model.glb --force-cpu
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from pipeline.orchestrator import Pipeline, PipelineConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """Load a PipelineConfig from a YAML config file.

    Reads the nested YAML structure and maps keys to PipelineConfig fields.
    Missing keys are handled gracefully with defaults.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A PipelineConfig populated from the YAML values.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    pipeline_cfg = data.get("pipeline", {})
    preprocessing = data.get("preprocessing", {})
    geometry = data.get("geometry", {})
    texturing = data.get("texturing", {})
    device = data.get("device", {})
    mesh_repair = data.get("mesh_repair", {})
    pbr = data.get("pbr", {})
    zero123 = data.get("zero123", {})
    exposure = data.get("exposure", {})
    shape_mv = data.get("shape_multiview", {})
    paint = data.get("paint", {})

    return PipelineConfig(
        backend=pipeline_cfg.get("backend", "hunyuan3d"),
        target_size=preprocessing.get("target_size", 512),
        remove_background=preprocessing.get("remove_background", True),
        geometry_seed=geometry.get("seed", 42),
        geometry_guidance_scale=geometry.get("guidance_scale", 7.5),
        geometry_steps=geometry.get("num_inference_steps", 50),
        texture_seed=texturing.get("seed", 42),
        texture_prompt=texturing.get("prompt", None),
        mesh_repair=mesh_repair.get("enabled", True),
        mesh_smooth_iterations=mesh_repair.get("smooth_iterations", 3),
        mesh_decimate_ratio=mesh_repair.get("decimate_ratio", None),
        generate_pbr=pbr.get("enabled", False),
        hunyuan3d_model_path=pipeline_cfg.get("hunyuan3d_model_path", None),
        skip_texturing=pipeline_cfg.get("skip_texturing", True),
        game_ready=pipeline_cfg.get("game_ready", True),
        game_ready_target_faces=pipeline_cfg.get("game_ready_target_faces", 50000),
        force_cpu=device.get("force_cpu", False),
        # Zero123++ multi-view
        zero123_steps=zero123.get("num_inference_steps", 75),
        zero123_guidance_scale=zero123.get("guidance_scale", 4.0),
        # Exposure correction
        correct_exposure=exposure.get("enabled", False),
        exposure_low_percentile=exposure.get("low_percentile", 1.0),
        exposure_high_percentile=exposure.get("high_percentile", 99.0),
        # Hunyuan3D-2mv shape
        shape_octree_resolution=shape_mv.get("octree_resolution", 384),
        # Hunyuan3D Paint
        paint_max_views=paint.get("max_views", 6),
        paint_resolution=paint.get("resolution", 512),
        paint_use_remesh=paint.get("use_remesh", True),
        mmgp_profile=paint.get("mmgp_profile", "LowRAM_LowVRAM"),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert 2D images to 3D shape models (.GLB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python run.py --input photo.png --output output/model.glb

  # Batch processing
  python run.py --batch-dir ./images/ --output-dir ./models/

  # Force CPU mode
  python run.py --input photo.png --output model.glb --force-cpu

  # Full-resolution mesh (skip game-ready decimation)
  python run.py --input photo.png --output model.glb --no-game-ready

  # Legacy textured pipeline
  python run.py --input photo.png --output model.glb --backend hi3dgen --prompt "medieval stone castle"
""",
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml if it exists)",
    )

    # Input/output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to a single input image (PNG/JPG/WEBP)",
    )
    input_group.add_argument(
        "--batch-dir", "-b",
        type=str,
        help="Directory containing multiple images for batch processing",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output .GLB file path (single image mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for batch processing (default: ./output)",
    )

    # Pipeline backend
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hi3dgen", "hunyuan3d", "triposg", "full"],
        default=None,
        help=(
            "Pipeline backend: 'hunyuan3d' (default) uses Hunyuan3D-2.1 fp16 "
            "shape generation; 'full' runs the 5-stage pipeline (Zero123++ -> "
            "Hunyuan3D-2mv -> mesh repair -> Hunyuan3D Paint -> PBR GLB); "
            "'triposg' uses TripoSG geometry; 'hi3dgen' preserves the legacy "
            "geometry+texture pipeline"
        ),
    )

    # Pipeline options
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU mode even if GPU is available",
    )
    parser.add_argument(
        "--skip-texturing",
        action="store_true",
        help="Skip texture generation (already enabled by default for Hunyuan3D-2.1)",
    )
    parser.add_argument(
        "--game-ready",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable game-ready decimation for the Hunyuan3D backend "
            "(use --no-game-ready to keep the full-resolution mesh)"
        ),
    )
    parser.add_argument(
        "--game-ready-target-faces",
        type=int,
        default=None,
        help="Target face count for game-ready Hunyuan export (default: 50000)",
    )
    parser.add_argument(
        "--no-bg-removal",
        action="store_true",
        help="Skip background removal preprocessing",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Text prompt for texture generation in the legacy textured pipeline",
    )

    # Quality settings
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="Target image size for preprocessing (default: 512)",
    )
    parser.add_argument(
        "--geometry-steps",
        type=int,
        default=50,
        help="Number of diffusion steps for geometry (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Full pipeline options
    parser.add_argument(
        "--correct-exposure",
        action="store_true",
        help="Apply dynamic range / exposure correction during preprocessing",
    )
    parser.add_argument(
        "--paint-resolution",
        type=int,
        default=None,
        help="Resolution for Hunyuan3D Paint texturing (default: 512)",
    )
    parser.add_argument(
        "--zero123-steps",
        type=int,
        default=None,
        help="Number of inference steps for Zero123++ multi-view generation (default: 75)",
    )
    parser.add_argument(
        "--octree-resolution",
        type=int,
        default=None,
        help="Octree resolution for Hunyuan3D-2mv shape generation (default: 384)",
    )
    parser.add_argument(
        "--mmgp-profile",
        type=str,
        choices=["LowRAM_LowVRAM", "LowRAM_HighVRAM", "HighRAM_LowVRAM", "HighRAM_HighVRAM"],
        default=None,
        help="MMGP offloading profile for Hunyuan3D Paint (default: LowRAM_LowVRAM)",
    )

    # External tool paths
    parser.add_argument(
        "--hi3dgen-path",
        type=str,
        default=None,
        help="Path to cloned Hi3DGen repository",
    )
    parser.add_argument(
        "--text2tex-path",
        type=str,
        default=None,
        help="Path to cloned Text2Tex repository",
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    return parser.parse_args()


def _cli_arg_was_provided(arg_name: str) -> bool:
    """Check whether a CLI argument was explicitly provided by the user.

    Scans sys.argv for the flag string (e.g. ``--target-size``) so we can
    distinguish "user passed --seed 42" from "argparse default is 42".
    """
    return any(a.startswith(arg_name) for a in sys.argv[1:])


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger(__name__)

    # -- Load YAML config (base layer) --
    yaml_path = args.config
    if yaml_path is None:
        # Auto-detect configs/default.yaml relative to this script
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir / "configs" / "default.yaml"
        if candidate.is_file():
            yaml_path = str(candidate)
            logger.info(f"Auto-detected config: {yaml_path}")

    if yaml_path is not None:
        logger.info(f"Loading config from: {yaml_path}")
        config = load_config_from_yaml(yaml_path)
    else:
        config = PipelineConfig()

    # -- Override with explicitly provided CLI args (highest priority) --
    if _cli_arg_was_provided("--backend"):
        config.backend = args.backend

    # store_true flags: if the user passed them, they are True.
    if args.force_cpu:
        config.force_cpu = True
    if args.skip_texturing:
        config.skip_texturing = True
    if _cli_arg_was_provided("--game-ready") or _cli_arg_was_provided("--no-game-ready"):
        config.game_ready = args.game_ready
    if args.no_bg_removal:
        config.remove_background = False

    # Value-based args: only override YAML when explicitly provided on CLI.
    if _cli_arg_was_provided("--target-size"):
        config.target_size = args.target_size
    if _cli_arg_was_provided("--geometry-steps"):
        config.geometry_steps = args.geometry_steps
    if _cli_arg_was_provided("--game-ready-target-faces"):
        if args.game_ready_target_faces < 4:
            logger.error("--game-ready-target-faces must be >= 4.")
            return 1
        config.game_ready = True
        config.game_ready_target_faces = args.game_ready_target_faces
    if _cli_arg_was_provided("--seed"):
        config.geometry_seed = args.seed
        config.texture_seed = args.seed
    if _cli_arg_was_provided("--prompt") or _cli_arg_was_provided("-p"):
        config.texture_prompt = args.prompt
    if _cli_arg_was_provided("--hi3dgen-path"):
        config.hi3dgen_path = args.hi3dgen_path
    if _cli_arg_was_provided("--text2tex-path"):
        config.text2tex_path = args.text2tex_path

    # Full pipeline overrides
    if args.correct_exposure:
        config.correct_exposure = True
    if _cli_arg_was_provided("--paint-resolution"):
        config.paint_resolution = args.paint_resolution
    if _cli_arg_was_provided("--zero123-steps"):
        config.zero123_steps = args.zero123_steps
    if _cli_arg_was_provided("--octree-resolution"):
        config.shape_octree_resolution = args.octree_resolution
    if _cli_arg_was_provided("--mmgp-profile"):
        config.mmgp_profile = args.mmgp_profile

    pipeline = Pipeline(config)

    if args.input:
        # Single image mode
        output = args.output
        if output is None:
            output = str(Path("./output") / f"{Path(args.input).stem}.glb")

        result = pipeline.process_image(args.input, output)

        if result.success:
            print(f"\nSUCCESS: {result.output_path}")
            print(f"  Vertices: {result.mesh_vertices}")
            print(f"  Faces: {result.mesh_faces}")
            print(f"  File size: {result.glb_size_mb:.1f} MB")
            print(f"  Duration: {result.duration_seconds:.1f}s")
            print(f"\nImport in Blender: File > Import > glTF 2.0 (.glb/.gltf)")
            return 0
        else:
            print(f"\nFAILED: {result.error}", file=sys.stderr)
            return 1

    elif args.batch_dir:
        # Batch mode — use --output-dir (--output is ignored in batch mode)
        output_dir = args.output_dir
        if args.output and not _cli_arg_was_provided("--output-dir"):
            logger.warning("--output is for single-image mode. Using --output-dir for batch. "
                           "Pass --output-dir explicitly for batch output location.")
            output_dir = args.output

        results = pipeline.process_batch(args.batch_dir, output_dir)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\nBATCH COMPLETE: {len(successful)}/{len(results)} succeeded")
        for r in successful:
            print(f"  OK: {r.output_path} ({r.glb_size_mb:.1f} MB)")
        for r in failed:
            print(f"  FAIL: {r.input_path} — {r.error}")

        if failed:
            return 1
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
