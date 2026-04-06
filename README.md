# 2D-to-3D Game Models

Current state of the project:

- **Default stable backend:** `Hunyuan3D-2.1` shape generation in `fp16`
- **Current SOTA textured backend:** `MV-Adapter -> Hunyuan3D-2mv -> Hunyuan3D Paint`
- **Target hardware:** Windows + CUDA, especially `RTX 3080 Ti / 16 GB`-class GPUs
- **Primary outputs:** Blender-ready `.glb` files, either geometry-only or textured PBR-style exports

The repo now supports two main workflows:

1. **Fast and reliable shape-only generation** using `Hunyuan3D-2.1`
2. **Higher-quality textured generation** using a full multi-stage pipeline with `MV-Adapter`, `Hunyuan3D-2mv`, and `Hunyuan3D Paint`

## Current Pipeline Modes

| Mode | Backend | Stack | Best For | 16 GB GPU Status |
| ---- | ------- | ----- | -------- | ---------------- |
| Stable geometry-only | `hunyuan3d` | Preprocess -> Hunyuan3D-2.1 -> optional decimation -> GLB | Fast iteration, mesh-only output | Best fit |
| SOTA textured | `full --multiview` | Preprocess -> MV-Adapter -> Hunyuan3D-2mv -> mesh prep -> Hunyuan3D Paint -> textured GLB | Best current textured result | Tuned for 16 GB |
| Safer textured fallback | `full --no-multiview` | Preprocess -> Hunyuan3D-2.1 -> mesh prep -> Hunyuan3D Paint -> textured GLB | When MV views are unstable or unnecessary | Lighter than multiview |
| Legacy comparison path | `hi3dgen` | Hi3DGen -> Text2Tex | Historical comparison only | Optional |

## What Is SOTA Here

For this repository, the most advanced path is the `full` backend with:

- `MV-Adapter` for high-resolution multiview image generation
- `Hunyuan3D-2mv` for multiview shape reconstruction
- `Hunyuan3D Paint` for texture generation
- GLB export with albedo + metallic + roughness when available

The default backend is still `hunyuan3d`, because it is the most stable and fastest workflow for everyday use.

## Quick Start

```bash
git clone https://github.com/pmikola/2d-to-3d-game-models.git
cd 2d-to-3d-game-models
pip install -r requirements.txt

# Clone the official Hunyuan3D-2.1 repo next to this project
cd ..
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd 2d-to-3d-game-models

# Stable default: geometry-only GLB
python run.py --input photo.png --output output/model.glb

# Current SOTA textured path
python run.py --input photo.png --output output/model_textured.glb --backend full --multiview
```

## Installation

### Tested Target

- Python `3.11`
- CUDA-enabled PyTorch
- Windows 11
- NVIDIA `RTX 3080 Ti Laptop GPU` with `16 GB` VRAM

### Core Project Dependencies

```bash
pip install -r requirements.txt
```

### Required Upstream Repo

Clone the official `Hunyuan3D-2.1` repository next to this project:

```bash
cd ..
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd 2d-to-3d-game-models
```

The runtime auto-detects a sibling `../Hunyuan3D-2.1` checkout. If you keep it elsewhere, set:

```bash
set HUNYUAN3D_21_REPO_PATH=C:\path\to\Hunyuan3D-2.1
```

or on PowerShell:

```powershell
$env:HUNYUAN3D_21_REPO_PATH = "C:\path\to\Hunyuan3D-2.1"
```

### Recommended Windows Extras

For the fastest Hunyuan Paint path on Windows, install:

- CUDA Toolkit
- Visual Studio Build Tools with C++ support

The project now auto-builds the Windows `custom_rasterizer` path when needed, but the compiled path is still the fast path.

### Optional Upstream Repos

These are not required for the main Hunyuan path:

- `Hi3DGen` for the legacy backend
- `Text2Tex` for the legacy texturing path
- `CharacterGen` as an alternative multiview generator

## Automatic First-Run Behavior

The project now handles much more of the setup automatically:

- Hunyuan model weights download from Hugging Face on first use
- `MV-Adapter` auto-installs from GitHub if missing
- `MV-Adapter` uses a `--no-deps` fallback to avoid unsupported optional packages such as `cvcuda_cu12`
- Hunyuan Paint auto-builds its Windows rasterizer path on first use
- If `realesrgan` / `basicsr` are unavailable on Windows, the pipeline falls back to PIL Lanczos upscaling instead of hard-failing
- If the compiled mesh inpaint extension is unavailable, the pipeline uses a simplified fallback instead of crashing

## Recommended Commands

### 1. Stable Geometry-Only Output

```bash
python run.py --input photo.png --output output/model.glb
```

This uses the default `hunyuan3d` backend and exports geometry only.

### 2. Highest-Quality Shape-Only Output

```bash
python run.py --input photo.png --output output/model_fullres.glb --backend hunyuan3d --no-game-ready --correct-exposure
```

Use this when you want the original higher-resolution shape instead of the default game-ready export.

### 3. Current SOTA Textured Output

Recommended starting point for `16 GB` GPUs:

```bash
python run.py --input photo.png --output output/model_textured.glb --backend full --multiview --multiview-generator mvadapter --correct-exposure --paint-resolution 512 --zero123-steps 35 --octree-resolution 512 --mmgp-profile LowRAM_LowVRAM --game-ready-target-faces 120000 --verbose
```

Notes:

- On `<18 GB` GPUs, the full backend now automatically:
  - caps Hunyuan Paint to `4` views
  - disables Paint remeshing
  - keeps `MMGP` enabled to avoid spilling into shared GPU memory
- Shared GPU memory works, but it is much slower than dedicated VRAM

### 4. Safer Single-View Textured Output

```bash
python run.py --input photo.png --output output/model_textured_single.glb --backend full --no-multiview --correct-exposure --paint-resolution 512 --mmgp-profile LowRAM_LowVRAM --game-ready-target-faces 120000 --verbose
```

Use this when:

- MV-Adapter views are poor
- you want to avoid the multiview stage entirely
- you need a lower-risk textured pipeline

### 5. Batch Processing

```bash
python run.py --batch-dir images --output-dir output
```

### 6. Paint-Only Smoke Test

Fast iteration for the current integrated Paint wrapper:

```bash
python test_paint.py --verbose --max-views 4 --no-remesh --export-glb
```

This runs the actual `pipeline.hunyuan3d_paint.Hunyuan3DPaintWrapper` path, not the older direct upstream import path.

### 7. Paint an Existing Mesh + Reference Image

```bash
python test_paint.py --mesh output\my_mesh.obj --image image_0001.png --max-views 4 --no-remesh --export-glb
```

Useful when:

- geometry is already generated
- you only want to iterate on Stage 4
- you want to keep testing inside dedicated VRAM

## Important Full-Backend Detail

In the `full` backend, the project always applies a practical mesh reduction step before texturing.

That is intentional: raw `Hunyuan3D-2mv` meshes can be too heavy for UV operations, texture baking, and GLB export. Use:

```bash
--game-ready-target-faces 120000
```

to preserve more detail while still keeping the texture stage tractable.

## Current Default Config

The shipped `configs/default.yaml` currently defaults to:

- `backend: hunyuan3d`
- `skip_texturing: true`
- `game_ready: true`
- `game_ready_target_faces: 120000`
- `use_multiview: false`
- `multiview_generator: mvadapter`
- `shape_multiview.octree_resolution: 512`
- `paint.resolution: 512`
- `paint.max_views: 6`
- `paint.use_remesh: true`
- `paint.mmgp_profile: LowRAM_LowVRAM`

Remember: the `full` backend overrides some of these automatically on `16 GB` GPUs to stay inside dedicated VRAM more often.

## Performance and VRAM Notes

### Shape-Only Backend

- Fastest workflow
- Best stability
- Best default for iterative 2D -> 3D shape generation

### Full Textured Backend

The heaviest stages are:

1. `MV-Adapter`
2. `Hunyuan3D Paint`

Current 16 GB tuning in the integrated path:

- `MV-Adapter` is chunked into two-view batches
- multiview steps are capped to `35` when needed
- Paint is run with `MMGP` on the internal multiview diffusion pipeline
- Paint is automatically reduced to `4` views on `<18 GB` GPUs
- Paint remesh is disabled on `<18 GB` GPUs to avoid shared-memory spill

If Task Manager shows heavy `shared GPU memory` usage:

- reduce `--game-ready-target-faces`
- keep `--paint-resolution 512`
- prefer `--no-multiview` if quality allows
- use `test_paint.py` to isolate paint before rerunning the full pipeline

## Output Files

### Geometry-Only Path

Typical outputs:

- `output/model.glb`
- `output/<stem>_preprocessed.png`

### Full Textured Path

Typical outputs:

- `output/model_textured.glb`
- `output/<stem>_preprocessed.png`
- `output/<stem>_view_front_az0.png`
- `output/<stem>_view_left_az270.png`
- `output/<stem>_view_back_az180.png`
- `output/<stem>_view_right_az90.png`

### Paint Smoke Test

Outputs are written under:

- `test_paint_tmp/paint_output/`
- optional exported GLB: `test_paint_tmp/paint_output.glb`

## Current Limitations

These are the main remaining realities of the project:

- Single-view 3D reconstruction is still ambiguous for backsides and hidden geometry
- `Hunyuan3D Paint` may return albedo + metallic + roughness but no normal map, depending on upstream output
- On Windows, `realesrgan` / `basicsr` may fall back to PIL upscaling
- The `full` backend is much slower than the default shape-only backend
- The legacy `hi3dgen` path is retained for comparison, not as the recommended path

## Are the Paint Outputs Supposed To Look Like UV Atlases

Yes.

The main albedo output from Hunyuan Paint is a UV texture image, not a rendered beauty image. For example:

- `textured.jpg` should look like a UV atlas
- `textured_roughness.jpg` should look like a grayscale roughness texture
- `textured_metallic.jpg` should look like a mostly dark metallic map for non-metal assets

What is **not** correct is exporting the saved `reference.png` as albedo. That bug has been fixed in the current wrapper.

## Troubleshooting

### `Hunyuan3D-2.1 paint code not found`

Clone the official repo next to this project or set `HUNYUAN3D_21_REPO_PATH`.

### `mvadapter` install fails

The project auto-installs it. If it still fails, check:

- internet access
- Git availability
- whether your environment blocks `pip` Git installs

### `custom_rasterizer` build issues on Windows

Install:

- CUDA Toolkit
- Visual Studio Build Tools with C++

The project now uses the Windows-specific rasterizer source tree and includes fixes for the previous 64-bit type mismatch on Windows.

### Paint is very slow and Task Manager shows shared GPU memory

That means the stage is spilling outside dedicated VRAM.

Try:

- `--paint-resolution 512`
- `--game-ready-target-faces 80000` or lower
- `--no-multiview`
- `python test_paint.py --max-views 4 --no-remesh --export-glb`

### Full backend shape is fine but textures are wrong

Check:

- the debug multiview images
- whether the mesh is too dense before paint
- whether the reference image is appropriate after preprocessing

### Need the full CLI flag list

Run:

```bash
python run.py --help
```

## Project Layout

```text
2d-to-3d-game-models/
├── run.py
├── test_paint.py
├── run_simple.py
├── configs/
│   └── default.yaml
├── pipeline/
│   ├── orchestrator.py
│   ├── hunyuan3d.py
│   ├── hunyuan3d_paint.py
│   ├── zero123plus.py
│   ├── preprocess.py
│   ├── mesh_repair.py
│   ├── export.py
│   └── device.py
├── requirements.txt
└── README.md
```

## Upstream Components

Main upstream projects used by this repository:

- [Tencent Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [MV-Adapter](https://github.com/huanngzh/MV-Adapter)
- [CharacterGen](https://github.com/zjpshadow/CharacterGen) (optional)
- [Hi3DGen](https://github.com/bytedance/Hi3DGen) (legacy)
- [Text2Tex](https://github.com/daveredrum/Text2Tex) (legacy)

## License Notes

This repository has its own license, but the models and upstream codebases it integrates are governed by their own licenses and usage terms.

Check the upstream repositories before using:

- Hunyuan3D-2.1
- MV-Adapter
- CharacterGen
- Hi3DGen
- Text2Tex
