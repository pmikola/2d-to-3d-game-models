#!/usr/bin/env python3
"""
2D-to-3D Game Model Generator

Converts a single 2D image (PNG/JPG) into a fully textured PBR 3D model (.GLB).

Pipeline:
  Stage 1: Hunyuan3D-2.1 — single-image 3D shape generation
  Stage 2: Hunyuan3D Paint — PBR texture painting (albedo, roughness, metallic)

Output is saved to output/<image_name>/ with both shape-only and textured GLB files.
Works on 16 GB VRAM (e.g. RTX 3080 Ti) via MMGP memory offloading.

Usage:
    python generate.py wizard.png
    python generate.py wizard.png --output output/wizard/wizard.glb
    python generate.py wizard.png --octree 512 --paint-views 6
    python generate.py wizard.png --no-texture
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve Hunyuan3D-2.1 repo root — look next to this project first, then
# common locations.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_CANDIDATES = [
    _SCRIPT_DIR.parent / "Hunyuan3D-2.1",          # sibling folder
    _SCRIPT_DIR / "Hunyuan3D-2.1",                  # child folder
    Path.home() / "Hunyuan3D-2.1",
    Path("/content/Hunyuan3D-2.1"),                  # Colab
]

HY3D_ROOT: Path | None = None
for p in _CANDIDATES:
    if (p / "hy3dshape").is_dir():
        HY3D_ROOT = p
        break

if HY3D_ROOT is None:
    print(
        "ERROR: Hunyuan3D-2.1 repo not found.\n"
        "Clone it next to this project:\n"
        "  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
    )
    sys.exit(1)

# Add to sys.path exactly as the official demo.py does.
sys.path.insert(0, str(HY3D_ROOT / "hy3dshape"))
sys.path.insert(0, str(HY3D_ROOT / "hy3dpaint"))

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
import torch
from PIL import Image

print(f"[init] Hunyuan3D-2.1 root: {HY3D_ROOT}")
print(f"[init] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"[init] GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")


def parse_args():
    ap = argparse.ArgumentParser(description="2D image → textured 3D GLB")
    ap.add_argument("image", type=str, help="Input image (PNG/JPG/WEBP)")
    ap.add_argument("--output", "-o", type=str, default=None,
                    help="Output GLB path (default: output/<stem>.glb)")
    ap.add_argument("--octree", type=int, default=512,
                    help="Octree resolution for shape (256-512, default 512)")
    ap.add_argument("--steps", type=int, default=50,
                    help="Diffusion steps for shape (default 50)")
    ap.add_argument("--paint-views", type=int, default=6,
                    help="Number of paint views (6-9, default 6)")
    ap.add_argument("--paint-resolution", type=int, default=512,
                    help="Paint resolution (512 or 768, default 512)")
    ap.add_argument("--no-texture", action="store_true",
                    help="Skip texture generation (geometry only)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def free_vram():
    """Aggressively free GPU memory between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"[vram] {alloc:.2f} GB allocated after cleanup")


# =========================================================================
# STAGE 1 — Shape Generation (~10 GB VRAM)
# =========================================================================
def run_shape(image_path: str, output_glb: str, octree: int, steps: int, seed: int):
    """Generate 3D mesh from a single image using Hunyuan3D-2.1."""
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    print(f"\n{'='*60}")
    print(f"[Stage 1] Shape generation  (octree={octree}, steps={steps})")
    print(f"{'='*60}")

    t0 = time.time()

    # Load image as RGBA (Hunyuan3D expects RGBA with proper alpha)
    image = Image.open(image_path)
    if image.mode == "RGB":
        print("[preprocess] Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)
    elif image.mode != "RGBA":
        image = image.convert("RGBA")

    print(f"[preprocess] Image: {image.size}, mode={image.mode}")

    # Load shape pipeline
    print("[shape] Loading Hunyuan3D-2.1 shape model...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Generate mesh
    print("[shape] Generating 3D geometry...")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    meshes = pipeline(
        image=image,
        num_inference_steps=steps,
        guidance_scale=5.0,
        octree_resolution=octree,
        num_chunks=20000,
        output_type="trimesh",
        generator=generator,
    )

    # Extract mesh (output is List[List[Trimesh]])
    mesh = meshes[0] if not isinstance(meshes[0], list) else meshes[0][0]
    mesh.export(output_glb)

    elapsed = time.time() - t0
    print(f"[shape] Done in {elapsed:.0f}s — {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"[shape] Saved: {output_glb}")

    # Free shape model VRAM
    del pipeline, meshes
    free_vram()

    return output_glb


# =========================================================================
# STAGE 2 — PBR Texture Generation (~14 GB VRAM with MMGP)
# =========================================================================
def run_paint(mesh_glb: str, image_path: str, output_glb: str,
              max_views: int, resolution: int):
    """Paint PBR textures onto the mesh using Hunyuan3D Paint + MMGP offloading."""
    print(f"\n{'='*60}")
    print(f"[Stage 2] PBR texture painting  (views={max_views}, res={resolution})")
    print(f"{'='*60}")

    t0 = time.time()

    # Provide a GPU-accelerated fallback for custom_rasterizer_kernel if the
    # CUDA C++ extension isn't compiled.  Uses a pure-PyTorch vectorized GPU
    # rasterizer — no C++ compilation or nvdiffrast required.
    try:
        import custom_rasterizer_kernel  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        import types as _t
        _crk = _t.ModuleType("custom_rasterizer_kernel")

        def _rasterize_image_pytorch(pos, tri, clamp_depth, width, height, eps, use_depth_prior):
            """Pure-PyTorch GPU rasterizer — no C++ compilation needed.

            Iterates over triangles in vectorized chunks.  Within each chunk
            the bounding-box pixel grid is tested against all chunk triangles
            via barycentric coordinates.  A unique-pixel sort resolves
            depth without any Python-level per-pixel loop.
            """
            device = pos.device
            num_faces = tri.shape[0]
            CHUNK = 512

            # -- Clip-space  ->  NDC  ->  screen-space (all vertices) ----------
            w_clip = pos[:, 3].clamp(min=eps)
            ndc_x = pos[:, 0] / w_clip
            ndc_y = pos[:, 1] / w_clip
            vdepth = pos[:, 2] / w_clip
            sx = (ndc_x * 0.5 + 0.5) * (width - 1)       # [V]
            sy = (ndc_y * 0.5 + 0.5) * (height - 1)       # [V]

            # -- Output buffers ------------------------------------------------
            zbuf     = torch.full((height * width,), 1e10, device=device)
            findices = torch.zeros(height * width, device=device, dtype=torch.int32)
            bary_out = torch.zeros(height * width, 3, device=device)

            for c0 in range(0, num_faces, CHUNK):
                c1   = min(c0 + CHUNK, num_faces)
                idx  = tri[c0:c1].long()                   # [C, 3]

                # Gather per-triangle screen coords & depth  [C, 3]
                vx = sx[idx];  vy = sy[idx];  vz = vdepth[idx]

                # Bounding boxes clamped to image
                bb_x0 = vx.min(dim=1).values.floor().clamp(min=0).long()
                bb_x1 = vx.max(dim=1).values.ceil().clamp(max=width  - 1).long()
                bb_y0 = vy.min(dim=1).values.floor().clamp(min=0).long()
                bb_y1 = vy.max(dim=1).values.ceil().clamp(max=height - 1).long()
                bb_w = (bb_x1 - bb_x0 + 1).clamp(min=0)
                bb_h = (bb_y1 - bb_y0 + 1).clamp(min=0)

                valid = (bb_w > 0) & (bb_h > 0)
                if not valid.any():
                    continue

                # Work only with valid triangles
                Cv = valid.sum().item()
                vx = vx[valid];  vy = vy[valid];  vz = vz[valid]
                bx0 = bb_x0[valid]; by0 = bb_y0[valid]
                bw  = bb_w[valid];   bh  = bb_h[valid]
                face_ids_chunk = (c0 + valid.nonzero(as_tuple=False).squeeze(1) + 1).int()

                max_w = bw.max().item();  max_h = bh.max().item()
                if max_w == 0 or max_h == 0:
                    continue

                # Local pixel grid  [max_h, max_w]
                gx = torch.arange(max_w, device=device, dtype=torch.float32)
                gy = torch.arange(max_h, device=device, dtype=torch.float32)
                gy, gx = torch.meshgrid(gy, gx, indexing="ij")

                # Absolute pixel coords  [Cv, max_h, max_w]
                px = gx.unsqueeze(0) + bx0.float().view(-1, 1, 1)
                py = gy.unsqueeze(0) + by0.float().view(-1, 1, 1)

                # Barycentric coordinates via edge functions
                ax = vx[:, 0].view(-1, 1, 1); ay = vy[:, 0].view(-1, 1, 1)
                bx_v = vx[:, 1].view(-1, 1, 1); by_v = vy[:, 1].view(-1, 1, 1)
                cx = vx[:, 2].view(-1, 1, 1); cy = vy[:, 2].view(-1, 1, 1)

                e10x = bx_v - ax;  e10y = by_v - ay
                e20x = cx   - ax;  e20y = cy   - ay
                det = (e10x * e20y - e20x * e10y).clamp(min=eps)

                dpx = px - ax;  dpy = py - ay
                w1 = (dpx * e20y - e20x * dpy) / det
                w2 = (e10x * dpy - dpx * e10y) / det
                w0 = 1.0 - w1 - w2

                # Mask: inside triangle AND within image bounds
                mask = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                        & (px >= 0) & (px < width)
                        & (py >= 0) & (py < height))
                if not mask.any():
                    continue

                # Interpolated depth at masked pixels
                pz = (w0 * vz[:, 0].view(-1, 1, 1)
                      + w1 * vz[:, 1].view(-1, 1, 1)
                      + w2 * vz[:, 2].view(-1, 1, 1))

                # Gather masked values  [N]
                tri_local = mask.nonzero(as_tuple=False)[:, 0]
                flat = (py[mask].long() * width + px[mask].long())
                z_vals = pz[mask]
                fids   = face_ids_chunk[tri_local]
                bw0 = w0[mask]; bw1 = w1[mask]; bw2 = w2[mask]

                # -- Vectorized z-buffer resolve (sort-based) ------------------
                # Sort by (flat_pixel, depth) so the nearest hit per pixel
                # comes first.  Then unique on flat_pixel keeps only winners.
                sort_key = flat.float() * 2e10 + z_vals
                order    = sort_key.argsort()
                flat_s   = flat[order]
                # First occurrence per pixel is the nearest
                uniq_mask = torch.ones(flat_s.shape[0], dtype=torch.bool, device=device)
                uniq_mask[1:] = flat_s[1:] != flat_s[:-1]

                win = order[uniq_mask]
                w_flat = flat[win]

                # Only update pixels where this chunk is closer than zbuf
                cur_z = zbuf[w_flat]
                closer = z_vals[win] < cur_z
                w_final = win[closer]
                pix     = flat[w_final]

                zbuf[pix]        = z_vals[w_final]
                findices[pix]    = fids[w_final]
                bary_out[pix, 0] = bw0[w_final]
                bary_out[pix, 1] = bw1[w_final]
                bary_out[pix, 2] = bw2[w_final]

            return (findices.view(height, width),
                    bary_out.view(height, width, 3))

        _crk.rasterize_image = _rasterize_image_pytorch
        sys.modules["custom_rasterizer_kernel"] = _crk

        # Also shim the top-level `custom_rasterizer` wrapper package that
        # MeshRender.py imports as `cr` and calls `cr.rasterize(pos, tri, res)`.
        import types as _tt
        _cr = _tt.ModuleType("custom_rasterizer")

        def _cr_rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
            assert pos.device == tri.device
            findices, barycentric = _crk.rasterize_image(
                pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
            )
            return findices, barycentric

        def _cr_interpolate(col, findices, barycentric, tri):
            f = findices - 1 + (findices == 0)
            vcol = col[0, tri.long()[f.long()]]
            result = barycentric.view(*barycentric.shape, 1) * vcol
            result = torch.sum(result, axis=-2)
            return result.view(1, *result.shape)

        _cr.rasterize = _cr_rasterize
        _cr.interpolate = _cr_interpolate
        sys.modules["custom_rasterizer"] = _cr
        print("[paint] custom_rasterizer + kernel shimmed — pure-PyTorch GPU rasterizer.")

    # Fix basicsr/realesrgan vs modern torchvision incompatibility.
    # basicsr imports torchvision.transforms.functional_tensor which was
    # removed in torchvision >=0.18.  Shim it back from the new location.
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        import torchvision.transforms.functional as _tvf
        import types
        _compat = types.ModuleType("torchvision.transforms.functional_tensor")
        _compat.rgb_to_grayscale = _tvf.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = _compat

    # Stub 'bpy' (Blender Python) — the Paint pipeline imports it at module
    # level in mesh_utils.py for an OBJ→GLB helper we don't use.  bpy is
    # only available inside Blender, so we create a lightweight fake module.
    import types
    if "bpy" not in sys.modules:
        _bpy = types.ModuleType("bpy")
        _bpy.data = types.ModuleType("bpy.data")
        _bpy.context = types.ModuleType("bpy.context")
        _bpy.ops = types.ModuleType("bpy.ops")
        _app = types.ModuleType("bpy.app")
        _app.version = (4, 0, 0)
        _bpy.app = _app
        sys.modules["bpy"] = _bpy
        sys.modules["bpy.data"] = _bpy.data
        sys.modules["bpy.context"] = _bpy.context
        sys.modules["bpy.ops"] = _bpy.ops
        sys.modules["bpy.app"] = _app

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    # Configure paint pipeline
    conf = Hunyuan3DPaintConfig(max_views, resolution)
    conf.multiview_cfg_path = str(HY3D_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = str(HY3D_ROOT / "hy3dpaint" / "hunyuanpaintpbr")

    # Auto-download RealESRGAN weights if missing (~67 MB)
    esrgan_path = HY3D_ROOT / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    esrgan_path.parent.mkdir(parents=True, exist_ok=True)
    if not esrgan_path.exists():
        esrgan_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        print(f"[paint] Downloading RealESRGAN weights (~67 MB)...")
        import urllib.request
        urllib.request.urlretrieve(esrgan_url, str(esrgan_path))
        print(f"[paint] Saved to: {esrgan_path}")
    conf.realesrgan_ckpt_path = str(esrgan_path)

    # MMGP offloading — keeps peak VRAM under 16 GB
    try:
        from mmgp import offload, profile_type
        print("[paint] Loading with MMGP offloading (LowRAM_LowVRAM)...")
        paint = Hunyuan3DPaintPipeline(conf)

        # MMGP expects a diffusers pipeline or dict of nn.Modules.
        # Hunyuan3DPaintPipeline is a custom wrapper — profile its internal
        # multiview diffusion pipeline (the heavy model) instead.
        mv_model = paint.models.get("multiview_model")
        if mv_model is not None and hasattr(mv_model, "pipeline"):
            offload.profile(mv_model.pipeline, profile_type.LowRAM_LowVRAM)
            print("[paint] MMGP applied to multiview diffusion pipeline.")
        else:
            # Fallback: try profiling the models dict directly
            offload.profile(paint.models, profile_type.LowRAM_LowVRAM)
            print("[paint] MMGP applied to paint models dict.")
    except ImportError:
        print("[paint] WARNING: mmgp not installed — trying without offloading.")
        print("[paint]   Install with: pip install mmgp")
        print("[paint]   Without MMGP, texture generation needs ~21 GB VRAM.")
        paint = Hunyuan3DPaintPipeline(conf)
    except Exception as mmgp_err:
        print(f"[paint] WARNING: MMGP offloading failed ({mmgp_err}). Running without it.")
        print("[paint]   This may cause OOM on GPUs < 24 GB.")

    # Generate PBR textures — Paint outputs an OBJ (not GLB).
    # We set save_glb=False because Paint's internal GLB converter needs
    # Blender's bpy module which we've stubbed.  We convert to GLB ourselves.
    import tempfile
    paint_tmp = tempfile.mkdtemp(prefix="paint_")
    paint_obj = os.path.join(paint_tmp, "textured.obj")

    print(f"[paint] Texturing mesh: {mesh_glb}")
    result_path = paint(
        mesh_path=mesh_glb,
        image_path=image_path,
        output_mesh_path=paint_obj,
        save_glb=False,
    )

    elapsed = time.time() - t0
    print(f"[paint] Texture generation done in {elapsed:.0f}s")
    print(f"[paint] OBJ output: {result_path}")

    # Convert the textured OBJ to GLB with embedded textures
    import trimesh
    print(f"[paint] Converting OBJ → GLB: {output_glb}")
    textured_mesh = trimesh.load(result_path, process=False)
    if isinstance(textured_mesh, trimesh.Scene):
        textured_mesh.export(output_glb, file_type="glb")
    else:
        textured_mesh.export(output_glb, file_type="glb")
    file_mb = os.path.getsize(output_glb) / (1024 * 1024)
    print(f"[paint] Saved GLB: {output_glb} ({file_mb:.1f} MB)")

    # Free paint model VRAM
    del paint
    free_vram()

    return result_path


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()

    # Verify input exists
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1

    # Create output folder: output/<image_name>/
    stem = Path(args.image).stem
    if args.output:
        out_dir = Path(args.output).parent
        final_glb = args.output
    else:
        out_dir = Path("output") / stem
        final_glb = str(out_dir / f"{stem}.glb")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[output] All results will be saved to: {out_dir}/")

    total_t0 = time.time()

    # Stage 1 — Shape
    shape_glb = str(out_dir / f"{stem}_shape.glb")
    if not args.no_texture:
        # When texturing, shape is intermediate; final GLB comes from paint
        pass
    else:
        final_glb = shape_glb

    run_shape(
        image_path=args.image,
        output_glb=shape_glb,
        octree=args.octree,
        steps=args.steps,
        seed=args.seed,
    )

    # Copy preprocessed image to output folder for reference
    preprocessed_src = Path("output") / f"{stem}_preprocessed.png"
    if preprocessed_src.exists() and preprocessed_src.parent != out_dir:
        import shutil
        shutil.copy2(preprocessed_src, out_dir / preprocessed_src.name)

    # Stage 2 — Paint (skip if --no-texture)
    if not args.no_texture:
        try:
            run_paint(
                mesh_glb=shape_glb,
                image_path=args.image,
                output_glb=final_glb,
                max_views=args.paint_views,
                resolution=args.paint_resolution,
            )
        except Exception as e:
            print(f"\n[paint] FAILED: {e}")
            print(f"[paint] The untextured shape is still available at: {shape_glb}")
            print(f"[paint] Common fixes:")
            print(f"  1. Install mmgp: pip install mmgp")
            print(f"  2. Compile custom_rasterizer: cd {HY3D_ROOT}/hy3dpaint/custom_rasterizer && pip install -e .")
            print(f"  3. Download RealESRGAN weights: check {HY3D_ROOT}/hy3dpaint/ckpt/")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[skip] Texture generation skipped (--no-texture)")

    total = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"COMPLETE in {total:.0f}s")
    print(f"Output folder: {out_dir}/")
    print(f"  Shape:    {shape_glb}")
    print(f"  Textured: {final_glb}")
    print(f"Import in Blender: File > Import > glTF 2.0 (.glb/.gltf)")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
