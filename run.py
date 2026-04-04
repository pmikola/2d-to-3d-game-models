#!/usr/bin/env python3
"""
2D-to-3D Game Models Pipeline — CLI Entry Point

Convert 2D images (PNG/JPG) to fully textured 3D models (.GLB) for Blender.

Two-stage SOTA pipeline:
  Stage 1: Hi3DGen (ICCV 2025) — high-fidelity 3D geometry from images
  Stage 2: Text2Tex / TEXTure — diffusion-based multi-view texture painting

Usage:
  python run.py --input photo.png --output model.glb
  python run.py --batch-dir ./images/ --output-dir ./models/
  python run.py --input photo.png --output model.glb --force-cpu
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.orchestrator import Pipeline, PipelineConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert 2D images to textured 3D models (.GLB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python run.py --input photo.png --output output/model.glb

  # Batch processing
  python run.py --batch-dir ./images/ --output-dir ./models/

  # Force CPU mode
  python run.py --input photo.png --output model.glb --force-cpu

  # Skip texturing (geometry only)
  python run.py --input photo.png --output model.glb --skip-texturing

  # Custom texture prompt
  python run.py --input photo.png --output model.glb --prompt "medieval stone castle"
""",
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
        help="Output path for single image (.GLB file) or output directory for batch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for batch processing (default: ./output)",
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
        help="Skip texture generation (output geometry-only GLB)",
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
        help="Text prompt for texture generation (auto-generated if not provided)",
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


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    # Build pipeline config
    config = PipelineConfig(
        target_size=args.target_size,
        remove_background=not args.no_bg_removal,
        geometry_seed=args.seed,
        geometry_steps=args.geometry_steps,
        texture_prompt=args.prompt,
        texture_seed=args.seed,
        hi3dgen_path=args.hi3dgen_path,
        text2tex_path=args.text2tex_path,
        force_cpu=args.force_cpu,
        skip_texturing=args.skip_texturing,
    )

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
        # Batch mode
        output_dir = args.output_dir
        if args.output:
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
