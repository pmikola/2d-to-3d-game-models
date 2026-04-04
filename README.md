# 2D-to-3D Game Models Pipeline

Convert 2D images (PNG/JPG) to fully textured 3D models (.GLB) — importable directly into Blender.

## How It Works

This is a two-stage SOTA pipeline:

| Stage | Model | What It Does |
|-------|-------|-------------|
| **1. Geometry** | [Hi3DGen](https://github.com/bytedance/Hi3DGen) (ICCV 2025, ByteDance) | Image → high-fidelity 3D mesh via NiRNE normal estimation + TRELLIS-based geometry diffusion |
| **2. Texturing** | [Text2Tex](https://github.com/daveredrum/Text2Tex) / [TEXTure](https://github.com/TEXTurePaper) | Bare mesh → textured mesh via depth-conditioned Stable Diffusion multi-view painting |

**Full pipeline:** Input image → background removal → Hi3DGen geometry → UV unwrapping → Text2Tex texturing → .GLB export

## Quick Start

```bash
# Clone
git clone https://github.com/pmikola/2d-to-3d-game-models.git
cd 2d-to-3d-game-models

# Install
pip install -r requirements.txt

# Run (single image)
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

### Step 2: Hi3DGen (Geometry)

```bash
# Clone Hi3DGen
git clone https://github.com/bytedance/Hi3DGen.git
cd Hi3DGen
pip install -r requirements.txt
cd ..

# Then point the pipeline to it:
python run.py --input photo.png --hi3dgen-path ./Hi3DGen/
```

The model weights (`Stable-X/trellis-normal-v0-1`) are auto-downloaded from HuggingFace on first run.

### Step 3: Text2Tex (Texturing) — Optional

```bash
# Clone Text2Tex
git clone https://github.com/daveredrum/Text2Tex.git
cd Text2Tex
pip install -r requirements.txt
cd ..

# Then point the pipeline to it:
python run.py --input photo.png --text2tex-path ./Text2Tex/
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

### Batch Processing

```bash
python run.py --batch-dir ./my_images/ --output-dir ./output/
```

### Custom Texture Prompt

```bash
python run.py --input photo.png -o model.glb --prompt "medieval stone castle, detailed PBR texture"
```

### Geometry Only (Skip Texturing)

```bash
python run.py --input photo.png -o model.glb --skip-texturing
```

### Force CPU Mode

```bash
python run.py --input photo.png -o model.glb --force-cpu
```

### All Options

```
usage: run.py [-h] (--input INPUT | --batch-dir BATCH_DIR)
              [--output OUTPUT] [--output-dir OUTPUT_DIR]
              [--force-cpu] [--skip-texturing] [--no-bg-removal]
              [--prompt PROMPT] [--target-size TARGET_SIZE]
              [--geometry-steps GEOMETRY_STEPS] [--seed SEED]
              [--hi3dgen-path HI3DGEN_PATH]
              [--text2tex-path TEXT2TEX_PATH]
              [--verbose]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input / -i` | Single input image path | — |
| `--batch-dir / -b` | Directory of images for batch | — |
| `--output / -o` | Output .GLB path (single) or dir (batch) | `./output/` |
| `--force-cpu` | Use CPU even if GPU available | `false` |
| `--skip-texturing` | Output geometry-only GLB | `false` |
| `--no-bg-removal` | Skip background removal | `false` |
| `--prompt / -p` | Texture generation prompt | Auto-generated |
| `--target-size` | Preprocessing resize target | `512` |
| `--geometry-steps` | Diffusion steps for geometry | `50` |
| `--seed` | Random seed | `42` |
| `--hi3dgen-path` | Path to Hi3DGen repo | — |
| `--text2tex-path` | Path to Text2Tex repo | — |
| `--verbose / -v` | Debug logging | `false` |

## Hardware Requirements

| Mode | VRAM | RAM | Speed (per image) |
|------|------|-----|--------------------|
| **GPU (recommended)** | 16+ GB | 16+ GB | ~2-5 minutes |
| **GPU (minimum)** | 8-16 GB | 16+ GB | ~5-10 minutes (reduced quality) |
| **CPU fallback** | — | 32+ GB | ~30-60 minutes |

The pipeline auto-detects your hardware and adjusts parameters:
- **16+ GB VRAM:** Full quality (36 viewpoints, 50 DDIM steps)
- **8-16 GB VRAM:** Reduced quality (18 viewpoints, 35 DDIM steps)
- **CPU:** Minimal quality (8 viewpoints, 25 DDIM steps)

## Import into Blender

1. Open Blender 4.x
2. **File → Import → glTF 2.0 (.glb/.gltf)**
3. Select the output `.glb` file
4. The model imports with geometry, UV mapping, and embedded textures

## Project Structure

```
2d-to-3d-game-models/
├── run.py                    # CLI entry point
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py       # Main pipeline: preprocess → geometry → texture → export
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

## SOTA Justification (2025-2026)

**Geometry — Hi3DGen (ICCV 2025):** Current state-of-the-art in open-source image-to-3D geometry generation. Uses NiRNE (Noise-injected Regressive Normal Estimator) for high-quality normal maps, then NoRLD (Normal-Regularized Latent Diffusion) with a TRELLIS-based backbone for geometry. Outperforms TRELLIS, TripoSG, Hunyuan3D, and CraftsMan in both professional and amateur user studies.

**Texturing — Text2Tex / TEXTure:** Multi-view diffusion-based texture painting using depth-conditioned Stable Diffusion 2. Progressively paints textures from multiple viewpoints (36 by default) with depth-aware inpainting, producing consistent, high-quality texture atlases. Represents the SOTA for applying textures to bare meshes using open-source diffusion models.

## Preprocessing

The pipeline automatically:
1. **Removes backgrounds** using [rembg](https://github.com/danielgatis/rembg) — Hi3DGen works best with isolated objects
2. **Resizes to 512x512** with aspect-ratio-preserving padding — matches Hi3DGen's internal resolution
3. **Converts formats** — handles PNG, JPG, JPEG, WEBP, BMP, TIFF
4. **Handles RGBA** — composites transparent images onto white background
5. **Checks quality** — warns on images that are too small (<256px) or too blurry

## Licenses

| Component | License |
|-----------|---------|
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
