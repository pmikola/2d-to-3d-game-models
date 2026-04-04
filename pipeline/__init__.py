"""
2D-to-3D Game Models Pipeline

Default pipeline for converting 2D images into geometry-only 3D models:
  Stage 1 (Geometry): Hunyuan3D-2.1 fp16 shape generation
  Stage 2 (Export): game-ready decimation, normalization, and GLB export

Legacy Hi3DGen + Text2Tex support remains available through the CLI backend switch.
"""

__version__ = "0.1.0"
