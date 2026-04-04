"""
Export module for converting textured meshes to .GLB format.

GLB (binary glTF 2.0) embeds geometry, UV coordinates, and textures into a single
file that is natively importable by Blender 4.x and online 3D viewers.
"""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def export_to_glb(
    mesh_path: str,
    texture_path: str | None,
    output_path: str,
) -> str:
    """
    Export a textured mesh to GLB format with embedded textures.

    Args:
        mesh_path: Path to the .OBJ mesh file.
        texture_path: Path to the texture image (PNG/JPG). If None, exports without texture.
        output_path: Path for the output .GLB file.

    Returns:
        Path to the exported .GLB file.
    """
    import numpy as np
    import trimesh

    logger.info(f"Exporting to GLB: {output_path}")
    logger.info(f"  Mesh: {mesh_path}")
    logger.info(f"  Texture: {texture_path}")

    # Load the mesh
    mesh = trimesh.load(mesh_path, process=False, force="mesh")

    if texture_path and Path(texture_path).exists():
        # Load texture image
        texture_image = Image.open(texture_path)
        logger.info(f"  Texture size: {texture_image.size}")

        # Create material with the texture
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_image,
            metallicFactor=0.0,
            roughnessFactor=0.8,
        )

        # Apply texture to mesh
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv,
                material=material,
            )
        else:
            logger.warning(
                "Mesh has no UV coordinates. Texture will not be applied correctly. "
                "Ensure UV unwrapping was performed during geometry generation."
            )
            # Try to apply anyway — trimesh may handle it
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
    else:
        logger.info("No texture provided — exporting geometry only.")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Export as GLB (binary glTF 2.0)
    mesh.export(str(output_file), file_type="glb")

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"GLB export complete: {output_file} ({file_size_mb:.1f} MB)")

    return str(output_file)


def export_textured_dir_to_glb(textured_dir: str, output_path: str) -> str:
    """
    Export a textured mesh directory (from Text2Tex output) to GLB.

    Looks for standard mesh + texture file patterns in the directory.

    Args:
        textured_dir: Directory containing textured .OBJ and texture image.
        output_path: Path for the output .GLB file.

    Returns:
        Path to the exported .GLB file.
    """
    textured_path = Path(textured_dir)

    # Find OBJ file
    obj_files = list(textured_path.glob("*.obj"))
    # Prefer textured version
    obj_path = None
    for obj in obj_files:
        if "textured" in obj.stem.lower():
            obj_path = obj
            break
    if obj_path is None and obj_files:
        obj_path = obj_files[0]

    if obj_path is None:
        raise FileNotFoundError(f"No .OBJ file found in {textured_dir}")

    # Find texture file
    texture_path = None
    texture_patterns = ["texture_atlas.*", "albedo.*", "diffuse.*", "material_0.*", "*.png", "*.jpg"]
    for pattern in texture_patterns:
        matches = list(textured_path.glob(pattern))
        # Filter to image files only
        image_matches = [
            m for m in matches
            if m.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and m.stem != obj_path.stem  # Don't pick up the OBJ filename
        ]
        if image_matches:
            texture_path = image_matches[0]
            break

    texture_str = str(texture_path) if texture_path else None
    return export_to_glb(str(obj_path), texture_str, output_path)


def validate_glb(glb_path: str) -> dict:
    """
    Validate a GLB file and return information about its contents.

    Args:
        glb_path: Path to the .GLB file.

    Returns:
        Dict with validation results and mesh statistics.
    """
    import trimesh

    path = Path(glb_path)
    if not path.exists():
        return {"valid": False, "error": f"File not found: {glb_path}"}

    try:
        scene = trimesh.load(glb_path)

        info = {
            "valid": True,
            "file_size_mb": path.stat().st_size / (1024 * 1024),
        }

        if isinstance(scene, trimesh.Scene):
            info["num_meshes"] = len(scene.geometry)
            total_verts = sum(len(g.vertices) for g in scene.geometry.values())
            total_faces = sum(len(g.faces) for g in scene.geometry.values())
            info["total_vertices"] = total_verts
            info["total_faces"] = total_faces
            info["has_textures"] = any(
                hasattr(g.visual, "material") and g.visual.material is not None
                for g in scene.geometry.values()
            )
        elif isinstance(scene, trimesh.Trimesh):
            info["num_meshes"] = 1
            info["total_vertices"] = len(scene.vertices)
            info["total_faces"] = len(scene.faces)
            info["has_textures"] = (
                hasattr(scene.visual, "material") and scene.visual.material is not None
            )

        logger.info(f"GLB validation: {info}")
        return info

    except Exception as e:
        return {"valid": False, "error": str(e)}
