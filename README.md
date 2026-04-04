# 2D-to-3D Game Models Pipeline

Convert 2D images (PNG/JPG) to geometry-only 3D models (.GLB) — importable directly into Blender.

The default profile now targets `Hunyuan3D-2.1` shape generation in `fp16`, then decimates the result to a game-ready mesh at about `50k` faces. It fits well on GPUs like the `RTX 3080 Ti 16GB` and exports an untextured shape model by default.

## How It Works

The default pipeline is shape-first and geometry-only:

| Stage | Model | What It Does |
| ----- | ----- | ------------ |
| **1. Geometry** | [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) fp16 | Image → high-fidelity 3D mesh using the official `hunyuan3d-dit-v2-1` shape checkpoint |
| **2. Export** | Built-in game-ready decimation + export pipeline | Decimate to a lower-poly mesh, normalize it, and export a geometry-only `.glb` |

**Default pipeline:** Input image → background removal → Hunyuan3D-2.1 shape generation → game-ready decimation (~50k faces) → normalization → geometry-only `.glb` export

**Legacy pipeline:** `--backend hi3dgen` keeps the older Hi3DGen + Text2Tex textured workflow available when needed.

## Quick Start

```bash
# Clone
git clone https://github.com/pmikola/2d-to-3d-game-models.git
cd 2d-to-3d-game-models

# Install project dependencies
pip install -r requirements.txt

# Clone the official Hunyuan3D-2.1 code next to the project
cd ..
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd 2d-to-3d-game-models

# Run (single image, geometry only + game-ready by default)
python run.py --input photo.png --output output/model.glb

# Run (batch)
python run.py --batch-dir ./images/ --output-dir ./models/
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA GPU with 16+ GB VRAM (recommended) or CPU (slower, but works)

### Step 1: Core Dependencies

```bash
pip install -r requirements.txt
```

This installs the Python packages this project needs for the default Hunyuan shape workflow.

### Step 2: Hunyuan3D-2.1 (Default Shape Generator)

Clone the official code repo next to this project:

```bash
cd ..
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd 2d-to-3d-game-models
```

The runtime auto-detects a local `../Hunyuan3D-2.1` checkout, or you can point to another location with the `HUNYUAN3D_21_REPO_PATH` environment variable.

The shipped config already points to the official Hugging Face weights:

- Repo: `tencent/Hunyuan3D-2.1`
- Shape checkpoint: `hunyuan3d-dit-v2-1`
- Precision on CUDA: `fp16`
- Default game-ready target: `50000` faces

The first run downloads the checkpoint automatically through the official loader.

### Step 3: Hi3DGen (Legacy Geometry Backend)

```bash
# Clone Hi3DGen
git clone https://github.com/bytedance/Hi3DGen.git
cd Hi3DGen
pip install -r requirements.txt
cd ..

# Then point the pipeline to it:
python run.py --input photo.png --backend hi3dgen --hi3dgen-path ./Hi3DGen/
```

The model weights (`Stable-X/trellis-normal-v0-1`) are auto-downloaded from HuggingFace on first run.

### Step 4: Text2Tex (Legacy Texturing) — Optional

```bash
# Clone Text2Tex
git clone https://github.com/daveredrum/Text2Tex.git
cd Text2Tex
pip install -r requirements.txt
cd ..

# Then point the pipeline to it:
python run.py --input photo.png --backend hi3dgen --text2tex-path ./Text2Tex/
```

If Text2Tex is not installed, the pipeline falls back to a built-in diffusers-based texturing approach.

### GPU Acceleration (Optional)

```bash
# For CUDA 12.4 (adjust version as needed)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.27.post2
pip install spconv-cu124==2.3.6
```

## Usage

### Single Image

```bash
python run.py --input photo.png --output output/model.glb
```

This uses `Hunyuan3D-2.1` fp16 and exports geometry only by default.

### Full-Resolution Mesh

```bash
python run.py --input photo.png --output output/model.glb --no-game-ready
```

Use this when you want the original high-resolution Hunyuan mesh instead of the default game-ready decimated export.

### Batch Processing

```bash
python run.py --batch-dir ./my_images/ --output-dir ./output/
```

### Legacy Textured Pipeline

```bash
python run.py --input photo.png -o model.glb --backend hi3dgen --prompt "medieval stone castle, detailed PBR texture"
```

### Geometry Only (Skip Texturing)

```bash
python run.py --input photo.png -o model.glb --skip-texturing
```

The default `Hunyuan3D-2.1` profile already behaves this way.

### Force CPU Mode

```bash
python run.py --input photo.png -o model.glb --force-cpu
```

### All Options

```text
usage: run.py [-h] (--input INPUT | --batch-dir BATCH_DIR)
              [--output OUTPUT] [--output-dir OUTPUT_DIR]
              [--force-cpu] [--skip-texturing]
              [--game-ready | --no-game-ready]
              [--game-ready-target-faces GAME_READY_TARGET_FACES]
              [--no-bg-removal] [--prompt PROMPT] [--target-size TARGET_SIZE]
              [--geometry-steps GEOMETRY_STEPS] [--seed SEED]
              [--hi3dgen-path HI3DGEN_PATH]
              [--text2tex-path TEXT2TEX_PATH]
              [--verbose]
```

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--input / -i` | Single input image path | — |
| `--batch-dir / -b` | Directory of images for batch | — |
| `--output / -o` | Output .GLB path (single) or dir (batch) | `./output/` |
| `--backend` | Default `hunyuan3d` shape-only backend or legacy `hi3dgen` | `hunyuan3d` |
| `--force-cpu` | Use CPU even if GPU available | `false` |
| `--skip-texturing` | Output geometry-only GLB | `true` for default config |
| `--game-ready / --no-game-ready` | Toggle Hunyuan game-ready decimation | `true` for default config |
| `--game-ready-target-faces` | Target face count for game-ready export | `50000` |
| `--no-bg-removal` | Skip background removal | `false` |
| `--prompt / -p` | Texture generation prompt for legacy pipeline | Auto-generated |
| `--target-size` | Preprocessing resize target | `512` |
| `--geometry-steps` | Diffusion steps for geometry | `50` |
| `--seed` | Random seed | `42` |
| `--hi3dgen-path` | Path to Hi3DGen repo | — |
| `--text2tex-path` | Path to Text2Tex repo | — |
| `--verbose / -v` | Debug logging | `false` |

## Hardware Requirements

| Mode | VRAM | RAM | Speed (per image) |
| ---- | ---- | --- | ----------------- |
| **GPU (default Hunyuan fp16)** | 10+ GB | 16+ GB | ~2-6 minutes |
| **GPU (comfortable)** | 16+ GB | 16+ GB | Best fit for the default profile |
| **CPU fallback** | — | 32+ GB | ~30-60 minutes |

The default Hunyuan game-ready profile fits well on an `RTX 3080 Ti 16GB`.

Legacy Hi3DGen + Text2Tex mode still auto-adjusts quality:

- **16+ GB VRAM:** Full quality (36 viewpoints, 50 DDIM steps)
- **8-16 GB VRAM:** Reduced quality (18 viewpoints, 35 DDIM steps)
- **CPU:** Minimal quality (8 viewpoints, 25 DDIM steps)

## Import into Blender

1. Open Blender 4.x
2. **File → Import → glTF 2.0 (.glb/.gltf)**
3. Select the output `.glb` file
4. The model imports with geometry only by default, and the Hunyuan profile is decimated for game-ready use unless you pass `--no-game-ready`

## Project Structure

```text
2d-to-3d-game-models/
├── run.py                    # CLI entry point
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py       # Main pipeline: preprocess → Hunyuan shape → game-ready export
│   ├── preprocess.py         # Image preprocessing (background removal, resize, etc.)
│   ├── geometry.py           # Hi3DGen wrapper for 3D mesh generation
│   ├── texturing.py          # Text2Tex / TEXTure wrapper for texture generation
│   ├── export.py             # Mesh export to .GLB with embedded textures
│   └── device.py             # GPU/CPU detection, VRAM checking, fallback logic
├── configs/
│   └── default.yaml          # Default pipeline parameters
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE                   # MIT
```

## Model Choice Notes

**Default backend — Hunyuan3D-2.1 fp16:** The project now defaults to the official Hunyuan shape generator because it provides a geometry-only workflow that fits well on `10-16 GB` GPUs, including an `RTX 3080 Ti 16GB`, and can be decimated into a more game-ready mesh automatically.

**Legacy backend — Hi3DGen + Text2Tex:** The older split geometry/texturing pipeline is still available with `--backend hi3dgen` when you want that workflow for comparison or experimentation.

## Preprocessing

The pipeline automatically:

1. **Removes backgrounds** using [rembg](https://github.com/danielgatis/rembg) — Hi3DGen works best with isolated objects
2. **Resizes to 512x512** with aspect-ratio-preserving padding — matches Hi3DGen's internal resolution
3. **Converts formats** — handles PNG, JPG, JPEG, WEBP, BMP, TIFF
4. **Handles RGBA** — composites transparent images onto white background
5. **Checks quality** — warns on images that are too small (<256px) or too blurry

## Licenses

| Component | License |
| --------- | ------- |
| This pipeline | MIT |
| Hi3DGen | MIT |
| rembg | MIT |
| trimesh | MIT |
| xatlas | MIT |
| Stable Diffusion 2 | OpenRAIL-M |
| Text2Tex | Check repo license |

## Troubleshooting

**"No CUDA GPU detected"** — Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

**"GPU out of memory"** — Try `--force-cpu`, close other GPU applications, or use a GPU with more VRAM.

**"rembg not installed"** — `pip install rembg` (CPU) or `pip install rembg[gpu]` (GPU)

**"xatlas not installed"** — `pip install xatlas` (required for UV unwrapping)

**"Hi3DGen not found"** — Clone it: `git clone https://github.com/bytedance/Hi3DGen.git` and pass `--hi3dgen-path ./Hi3DGen/`
