#!/usr/bin/env python3
"""
Smoke-test the current Hunyuan Paint integration in isolation.

This test exercises the project's *current* integrated path:
`pipeline.hunyuan3d_paint.Hunyuan3DPaintWrapper`

By default it creates:
1. a small dummy OBJ mesh
2. a simple dummy reference image
3. a paint-only output directory

Unlike the older version of this script, it does not bypass the wrapper and
does not import Tencent's raw pipeline directly. That makes it useful for fast
iteration on the actual code path used by `pipeline.orchestrator`.

Examples:
  python test_paint.py
  python test_paint.py --export-glb
  python test_paint.py --mesh output/my_mesh.obj --image image_0001.png --export-glb
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.device import detect_device, release_runtime_memory
from pipeline.export import export_hunyuan_paint_to_glb
from pipeline.hunyuan3d_paint import Hunyuan3DPaintWrapper


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test_paint_tmp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the integrated Hunyuan Paint stage.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Optional existing OBJ/GLB mesh to texture. If omitted, a dummy OBJ is created.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional reference image path. If omitted, a dummy image is created.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for test assets and paint outputs.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Paint resolution (default: 512).",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=6,
        help="Max paint views (default: 6).",
    )
    parser.add_argument(
        "--mmgp-profile",
        type=str,
        default="LowRAM_LowVRAM",
        choices=[
            "LowRAM_LowVRAM",
            "LowRAM_HighVRAM",
            "HighRAM_LowVRAM",
            "HighRAM_HighVRAM",
        ],
        help="MMGP profile for the paint wrapper.",
    )
    parser.add_argument(
        "--no-remesh",
        action="store_true",
        help="Disable Hunyuan Paint remeshing for this smoke test.",
    )
    parser.add_argument(
        "--export-glb",
        action="store_true",
        help="Also export the paint result to a GLB using the project export path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_dummy_mesh(mesh_path: Path) -> Path:
    import trimesh

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh = mesh.subdivide().subdivide()
    mesh.export(mesh_path)
    print(
        f"[test] Created dummy mesh: {mesh_path} "
        f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)"
    )
    return mesh_path


def create_dummy_image(image_path: Path, size: int = 512) -> Path:
    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((96, 72, 416, 440), radius=48, fill=(72, 122, 214))
    draw.ellipse((150, 118, 362, 298), fill=(245, 205, 84))
    draw.rectangle((188, 290, 324, 388), fill=(181, 91, 52))
    draw.rectangle((140, 396, 372, 436), fill=(54, 54, 54))
    draw.line((96, 256, 416, 256), fill=(255, 255, 255), width=10)

    image.save(image_path)
    print(f"[test] Created dummy image: {image_path} ({size}x{size})")
    return image_path


def resolve_assets(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    mesh_path = Path(args.mesh) if args.mesh else output_dir / "dummy_cube.obj"
    image_path = Path(args.image) if args.image else output_dir / "dummy_reference.png"

    if args.mesh is None:
        create_dummy_mesh(mesh_path)
    elif not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    else:
        print(f"[test] Using existing mesh: {mesh_path}")

    if args.image is None:
        create_dummy_image(image_path, size=args.resolution)
    elif not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    else:
        print(f"[test] Using existing image: {image_path}")

    return mesh_path, image_path


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path, image_path = resolve_assets(args, output_dir)
    texture_dir = output_dir / "paint_output"
    output_glb = output_dir / "paint_output.glb"

    device_config = detect_device()
    paint = Hunyuan3DPaintWrapper(
        device_config=device_config,
        model_path="tencent/Hunyuan3D-2.1",
        max_views=args.max_views,
        resolution=args.resolution,
        mmgp_profile=args.mmgp_profile,
    )

    print("[test] Running integrated paint smoke test...")
    print(f"[test] Mesh: {mesh_path}")
    print(f"[test] Image: {image_path}")
    print(
        f"[test] Settings: resolution={args.resolution}, "
        f"views={args.max_views}, use_remesh={not args.no_remesh}"
    )

    t0 = time.time()
    paint_result = None
    try:
        reference_image = Image.open(image_path).convert("RGB")
        paint_result = paint.generate_textures(
            mesh_path=str(mesh_path),
            reference_image=reference_image,
            output_dir=str(texture_dir),
            use_remesh=not args.no_remesh,
        )
        elapsed = time.time() - t0
        print(f"[test] Paint succeeded in {elapsed:.1f}s")
        print(f"[test] Paint output dir: {texture_dir}")
        for key, value in paint_result.items():
            print(f"[test] {key}: {value}")

        if args.export_glb:
            export_hunyuan_paint_to_glb(
                textured_obj_dir=str(texture_dir),
                output_path=str(output_glb),
                texture_paths=paint_result,
            )
            print(f"[test] Exported GLB: {output_glb}")

        return 0
    finally:
        paint.unload()
        release_runtime_memory("after_test_paint")


if __name__ == "__main__":
    raise SystemExit(main())
