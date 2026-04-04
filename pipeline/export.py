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
    normal_map_path: str | None = None,
    roughness_map_path: str | None = None,
    metallic_map_path: str | None = None,
) -> str:
    """
    Export a textured mesh to GLB format with embedded textures and PBR maps.

    Args:
        mesh_path: Path to the .OBJ mesh file.
        texture_path: Path to the albedo texture image (PNG/JPG). If None, exports without texture.
        output_path: Path for the output .GLB file.
        normal_map_path: Path to normal map image. If None, omitted from material.
        roughness_map_path: Path to roughness map image. If None, omitted from material.
        metallic_map_path: Path to metallic map image. If None, omitted from material.

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

        # Build PBR material kwargs
        pbr_kwargs = {
            "baseColorTexture": texture_image,
            "metallicFactor": 0.0,
            "roughnessFactor": 0.8,
        }

        # Add normal map if available (standard glTF 2.0 tangent-space normal)
        if normal_map_path and Path(normal_map_path).exists():
            pbr_kwargs["normalTexture"] = Image.open(normal_map_path)
            logger.info(f"  Normal map: {normal_map_path}")

        # Build combined ORM metallicRoughnessTexture per glTF 2.0 spec:
        #   R = occlusion (255 = full, no AO), G = roughness, B = metallic
        has_roughness = roughness_map_path and Path(roughness_map_path).exists()
        has_metallic = metallic_map_path and Path(metallic_map_path).exists()
        if has_roughness or has_metallic:
            # Determine texture dimensions from whichever map is available
            ref_img = Image.open(roughness_map_path) if has_roughness else Image.open(metallic_map_path)
            tex_w, tex_h = ref_img.size

            roughness_arr = np.array(ref_img.convert("L")) if has_roughness else np.full((tex_h, tex_w), 204, dtype=np.uint8)
            if has_metallic:
                metallic_img = Image.open(metallic_map_path).convert("L").resize((tex_w, tex_h), Image.LANCZOS)
                metallic_arr = np.array(metallic_img)
            else:
                metallic_arr = np.zeros((tex_h, tex_w), dtype=np.uint8)

            # ORM: R=occlusion(white=1.0), G=roughness, B=metallic
            occlusion_arr = np.full((tex_h, tex_w), 255, dtype=np.uint8)
            orm_array = np.stack([occlusion_arr, roughness_arr, metallic_arr], axis=-1)
            orm_texture = Image.fromarray(orm_array, mode="RGB")

            pbr_kwargs["metallicRoughnessTexture"] = orm_texture
            pbr_kwargs["metallicFactor"] = 1.0
            pbr_kwargs["roughnessFactor"] = 1.0
            logger.info(f"  Combined ORM metallicRoughnessTexture: {tex_w}x{tex_h}")
            if has_roughness:
                logger.info(f"  Roughness map: {roughness_map_path}")
            if has_metallic:
                logger.info(f"  Metallic map: {metallic_map_path}")

        # Create PBR material
        material = trimesh.visual.material.PBRMaterial(**pbr_kwargs)

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

    # Look for PBR maps in a 'pbr' subdirectory
    pbr_dir = textured_path / "pbr"
    normal_map = None
    roughness_map = None
    metallic_map = None
    if pbr_dir.is_dir():
        for f in pbr_dir.glob("normal_map.*"):
            normal_map = str(f)
        for f in pbr_dir.glob("roughness_map.*"):
            roughness_map = str(f)
        for f in pbr_dir.glob("metallic_map.*"):
            metallic_map = str(f)

    return export_to_glb(
        str(obj_path),
        texture_str,
        output_path,
        normal_map_path=normal_map,
        roughness_map_path=roughness_map,
        metallic_map_path=metallic_map,
    )


def export_hunyuan_paint_to_glb(textured_obj_dir: str, output_path: str, texture_paths: dict) -> str:
    """Export Hunyuan3D Paint output to GLB with correct PBR materials.

    Locates the OBJ and texture files produced by Hunyuan3D Paint in the given
    directory and calls ``export_to_glb`` with the right PBR map paths.

    Args:
        textured_obj_dir: Directory containing Paint output (OBJ + textures).
        output_path: Destination path for the final GLB file.
        texture_paths: Dict returned by ``Hunyuan3DPaintWrapper.generate_textures()``
            with keys: textured_obj, albedo, normal, roughness, metallic.

    Returns:
        Path to the exported GLB file.
    """
    obj_dir = Path(textured_obj_dir)

    # Resolve OBJ mesh path
    mesh_path = texture_paths.get("textured_obj")
    if not mesh_path or not Path(mesh_path).exists():
        # Fallback: scan directory for an OBJ
        obj_files = list(obj_dir.rglob("*.obj"))
        if not obj_files:
            raise FileNotFoundError(f"No OBJ file found in {textured_obj_dir}")
        mesh_path = str(obj_files[0])
    logger.info(f"Exporting Paint output to GLB: mesh={mesh_path}")

    return export_to_glb(
        mesh_path=mesh_path,
        texture_path=texture_paths.get("albedo"),
        output_path=output_path,
        normal_map_path=texture_paths.get("normal"),
        roughness_map_path=texture_paths.get("roughness"),
        metallic_map_path=texture_paths.get("metallic"),
    )


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
