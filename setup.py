"""Setup script for 2D-to-3D Game Models Pipeline."""

from setuptools import find_packages, setup

setup(
    name="image-to-3d-pipeline",
    version="0.1.0",
    description="Convert 2D images to fully textured PBR 3D models (.GLB) using Hunyuan3D-2.1",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="2D-to-3D Pipeline Contributors",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "trimesh>=4.0",
        "pygltflib>=1.16",
        "xatlas>=0.0.9",
        "Pillow>=10.0",
        "rembg>=2.0",
        "numpy>=1.24",
        "diffusers>=0.14.0",
        "transformers>=4.27.4",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "huggingface_hub>=0.20.0",
        "einops",
        "omegaconf",
        "pytorch-lightning",
        "torchdiffeq",
        "pymeshlab",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "gpu": [
            "xformers>=0.0.27",
        ],
        "demo": [
            "gradio>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "image-to-3d=generate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
)
