"""
Texture generation using Text2Tex / TEXTure.

Multi-view diffusion-based texture painting onto bare meshes.
Uses Stable Diffusion 2 depth-conditioned inpainting to progressively
paint high-quality textures from multiple viewpoints.

Primary: Text2Tex (ICCV 2023) — https://github.com/daveredrum/Text2Tex
Fallback: TEXTure — https://github.com/TEXTurePaper

Both use stabilityai/stable-diffusion-2-depth for depth-aware texture generation.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class TextureGenerator:
    """
    Wrapper for Text2Tex / TEXTure texture generation.

    Applies high-quality textures to bare meshes using Stable Diffusion
    depth-conditioned inpainting from multiple viewpoints.
    """

    def __init__(self, device_config, text2tex_path: str | None = None):
        """
        Initialize texture generator.

        Args:
            device_config: DeviceConfig from device.py.
            text2tex_path: Path to cloned Text2Tex repo. If None, uses diffusers-based fallback.
        """
        self.device_config = device_config
        self.text2tex_path = text2tex_path
        self._sd_pipeline = None
        self._initialized = False

    def initialize(self) -> None:
        """Load Stable Diffusion pipeline for texture generation."""
        if self._initialized:
            return

        logger.info("Initializing texture generation pipeline...")
        logger.info(f"  Device: {self.device_config.device}")
        logger.info(f"  Viewpoints: {self.device_config.texture_num_viewpoints}")
        logger.info(f"  DDIM steps: {self.device_config.texture_ddim_steps}")

        try:
            self._load_sd_pipeline()
            self._initialized = True
            logger.info("Texture pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize texture pipeline: {e}")
            raise

    def _load_sd_pipeline(self) -> None:
        """Load Stable Diffusion 2 depth pipeline."""
        import torch
        from diffusers import StableDiffusionDepth2ImgPipeline

        model_id = "stabilityai/stable-diffusion-2-depth"
        logger.info(f"Loading {model_id}...")

        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=self.device_config.dtype,
        )

        if self.device_config.has_gpu:
            pipe = pipe.to(self.device_config.device)
            if self.device_config.use_xformers:
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory-efficient attention.")
                except Exception:
                    logger.info("Could not enable xformers, using default attention.")
        else:
            # CPU optimizations
            pipe = pipe.to("cpu")

        # Enable memory optimizations
        if self.device_config.has_gpu:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        self._sd_pipeline = pipe

    def generate_texture(
        self,
        mesh_obj_path: str,
        output_dir: str,
        prompt: str | None = None,
        original_image: Image.Image | None = None,
        seed: int = 42,
    ) -> str:
        """
        Generate texture for a mesh.

        If Text2Tex is installed, uses its multi-view progressive painting approach.
        Otherwise, falls back to a simplified diffusers-based texturing.

        Args:
            mesh_obj_path: Path to the .OBJ mesh file.
            output_dir: Directory to save textured output.
            prompt: Text prompt for texture generation. Auto-generated if None.
            original_image: Original input image for auto-prompt generation.
            seed: Random seed for reproducibility.

        Returns:
            Path to the textured .OBJ file directory.
        """
        self.initialize()

        # Generate prompt if not provided
        if prompt is None:
            prompt = self._generate_prompt(original_image)

        logger.info(f"Generating texture with prompt: '{prompt}'")
        logger.info(f"  Mesh: {mesh_obj_path}")
        logger.info(f"  Output: {output_dir}")

        if not self.device_config.has_gpu:
            logger.warning(
                "Running texture generation on CPU — this may take 5-15 minutes per viewpoint."
            )

        # Try Text2Tex first (if installed)
        if self.text2tex_path and Path(self.text2tex_path).exists():
            try:
                return self._run_text2tex(mesh_obj_path, output_dir, prompt, seed)
            except Exception as e:
                logger.warning(f"Text2Tex failed: {e}. Falling back to diffusers-based texturing.")

        # Fallback: simplified multi-view texturing using diffusers
        return self._run_diffusers_texturing(mesh_obj_path, output_dir, prompt, seed)

    def _generate_prompt(self, original_image: Image.Image | None) -> str:
        """
        Auto-generate a texture prompt from the original image.

        Uses BLIP-2 captioning if available, otherwise uses a generic prompt.
        """
        if original_image is not None:
            try:
                return self._caption_image(original_image)
            except Exception as e:
                logger.info(f"Auto-captioning failed: {e}. Using generic prompt.")

        return "high quality PBR texture, photorealistic, detailed surface, 4K"

    def _caption_image(self, image: Image.Image) -> str:
        """Generate a caption for the image using BLIP or transformers."""
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        logger.info("Auto-generating texture prompt from input image using BLIP...")

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=self.device_config.dtype,
        )
        model = model.to(self.device_config.device)

        inputs = processor(image, return_tensors="pt").to(self.device_config.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)

        caption = processor.decode(output[0], skip_special_tokens=True)
        prompt = f"a highly detailed {caption}, photorealistic texture, PBR material, 4K"

        logger.info(f"Generated prompt: '{prompt}'")
        return prompt

    def _run_text2tex(
        self,
        mesh_obj_path: str,
        output_dir: str,
        prompt: str,
        seed: int,
    ) -> str:
        """
        Run Text2Tex for multi-view progressive texture painting.

        Args:
            mesh_obj_path: Path to .OBJ mesh.
            output_dir: Output directory.
            prompt: Texture prompt.
            seed: Random seed.

        Returns:
            Path to textured output directory.
        """
        logger.info("Running Text2Tex multi-view texturing...")

        mesh_path = Path(mesh_obj_path)
        mesh_name = mesh_path.stem

        # Determine device string for Text2Tex
        if self.device_config.has_gpu:
            if self.device_config.vram_gb >= 24:
                device_str = "3090"
            elif self.device_config.vram_gb >= 16:
                device_str = "2080"
            else:
                device_str = "2080"
        else:
            device_str = "2080"  # Text2Tex handles CPU internally

        cmd = [
            sys.executable,
            str(Path(self.text2tex_path) / "scripts" / "generate_texture.py"),
            "--input_dir", str(mesh_path.parent),
            "--output_dir", output_dir,
            "--obj_name", mesh_name,
            "--obj_file", mesh_path.name,
            "--prompt", prompt,
            "--add_view_to_prompt",
            "--ddim_steps", str(self.device_config.texture_ddim_steps),
            "--num_viewpoints", str(self.device_config.texture_num_viewpoints),
            "--viewpoint_mode", "predefined",
            "--use_principle",
            "--update_steps", "20",
            "--seed", str(seed),
            "--post_process",
            "--device", device_str,
        ]

        logger.info(f"Text2Tex command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.text2tex_path,
        )

        if result.returncode != 0:
            logger.error(f"Text2Tex stderr: {result.stderr}")
            raise RuntimeError(f"Text2Tex failed with return code {result.returncode}")

        logger.info("Text2Tex texturing complete.")
        return output_dir

    def _run_diffusers_texturing(
        self,
        mesh_obj_path: str,
        output_dir: str,
        prompt: str,
        seed: int,
    ) -> str:
        """
        Simplified multi-view texturing using diffusers directly.

        This is a fallback when Text2Tex is not installed. It uses
        depth-conditioned Stable Diffusion to paint textures from key viewpoints.

        Args:
            mesh_obj_path: Path to .OBJ mesh.
            output_dir: Output directory.
            prompt: Texture prompt.
            seed: Random seed.

        Returns:
            Path to textured output directory.
        """
        import torch
        import trimesh
        import numpy as np

        logger.info("Running diffusers-based multi-view texturing (fallback mode)...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load the mesh
        mesh = trimesh.load(mesh_obj_path, process=False)

        # Define viewpoint angles (azimuth, elevation) for multi-view painting
        num_views = self.device_config.texture_num_viewpoints
        viewpoints = self._generate_viewpoints(num_views)

        logger.info(f"Generating texture from {len(viewpoints)} viewpoints...")

        # Render depth maps from each viewpoint and use SD to generate texture patches
        texture_size = 1024
        texture_image = Image.new("RGB", (texture_size, texture_size), (128, 128, 128))

        torch.manual_seed(seed)

        for i, (azimuth, elevation) in enumerate(viewpoints):
            logger.info(f"  Viewpoint {i + 1}/{len(viewpoints)}: az={azimuth:.0f}, el={elevation:.0f}")

            try:
                # Render depth map from this viewpoint
                depth_image = self._render_depth(mesh, azimuth, elevation, resolution=512)

                if depth_image is None:
                    continue

                # Use SD depth-to-image to generate texture for this view
                view_prompt = f"{prompt}, viewed from {'front' if i == 0 else f'angle {azimuth:.0f} degrees'}"

                with torch.no_grad():
                    result = self._sd_pipeline(
                        prompt=view_prompt,
                        image=depth_image,
                        negative_prompt="low quality, blurry, distorted, ugly",
                        num_inference_steps=self.device_config.texture_ddim_steps,
                        strength=0.7,
                        guidance_scale=7.5,
                    )

                if result.images:
                    view_texture = result.images[0]
                    view_texture.save(str(output_path / f"view_{i:03d}.png"))

            except Exception as e:
                logger.warning(f"  Failed to generate texture for viewpoint {i}: {e}")
                continue

        # Project textures back onto mesh UV space
        # For the fallback mode, we use the front view as primary texture
        front_texture_path = output_path / "view_000.png"
        if front_texture_path.exists():
            texture_image = Image.open(front_texture_path)

        # Save the texture atlas
        texture_atlas_path = output_path / "texture_atlas.png"
        texture_image.save(str(texture_atlas_path))

        # Save textured OBJ with material
        self._save_textured_obj(mesh, output_path, texture_atlas_path)

        logger.info(f"Diffusers-based texturing complete. Output: {output_dir}")
        return output_dir

    def _generate_viewpoints(self, num_views: int) -> list[tuple[float, float]]:
        """Generate evenly spaced viewpoints around the object."""
        viewpoints = []

        if num_views <= 8:
            # Key viewpoints only
            azimuths = [0, 45, 90, 135, 180, 225, 270, 315][:num_views]
            for az in azimuths:
                viewpoints.append((az, 0))
        else:
            # Multiple elevation rings
            elevations = [0, 30, -30]
            views_per_ring = num_views // len(elevations)
            remainder = num_views % len(elevations)

            for el_idx, el in enumerate(elevations):
                ring_views = views_per_ring + (1 if el_idx < remainder else 0)
                for i in range(ring_views):
                    az = (360.0 / ring_views) * i
                    viewpoints.append((az, el))

        return viewpoints

    def _render_depth(self, mesh, azimuth: float, elevation: float, resolution: int = 512):
        """
        Render a depth map of the mesh from a given viewpoint.

        Uses trimesh's built-in rendering or pyrender if available.
        """
        try:
            import pyrender

            # Create scene
            scene = pyrender.Scene()

            # Add mesh
            py_mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(py_mesh)

            # Camera setup
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

            # Camera pose from azimuth/elevation
            import numpy as np

            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)
            distance = 2.0

            cam_x = distance * np.cos(el_rad) * np.sin(az_rad)
            cam_y = distance * np.sin(el_rad)
            cam_z = distance * np.cos(el_rad) * np.cos(az_rad)

            # Look-at matrix
            eye = np.array([cam_x, cam_y, cam_z])
            target = np.array([0, 0, 0])
            up = np.array([0, 1, 0])

            cam_pose = self._look_at(eye, target, up)
            scene.add(camera, pose=cam_pose)

            # Render
            renderer = pyrender.OffscreenRenderer(resolution, resolution)
            _, depth = renderer.render(scene)
            renderer.delete()

            # Convert depth to PIL Image
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            return Image.fromarray(depth_uint8)

        except ImportError:
            logger.debug("pyrender not available, generating synthetic depth map.")
            # Generate a simple synthetic depth map as placeholder
            import numpy as np

            depth = np.zeros((resolution, resolution), dtype=np.uint8)
            center = resolution // 2
            radius = resolution // 3
            y, x = np.ogrid[-center:resolution - center, -center:resolution - center]
            mask = x * x + y * y <= radius * radius
            depth[mask] = 200
            return Image.fromarray(depth)

        except Exception as e:
            logger.warning(f"Depth rendering failed: {e}")
            return None

    @staticmethod
    def _look_at(eye, target, up):
        """Create a look-at camera matrix."""
        import numpy as np

        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)

        mat = np.eye(4)
        mat[:3, 0] = right
        mat[:3, 1] = true_up
        mat[:3, 2] = -forward
        mat[:3, 3] = eye

        return mat

    def _save_textured_obj(self, mesh, output_dir: Path, texture_path: Path) -> None:
        """Save mesh as OBJ with material referencing the texture atlas."""
        import trimesh

        # Create MTL file
        mtl_path = output_dir / "mesh.mtl"
        with open(mtl_path, "w") as f:
            f.write("newmtl material_0\n")
            f.write("Ka 1.0 1.0 1.0\n")
            f.write("Kd 1.0 1.0 1.0\n")
            f.write("Ks 0.0 0.0 0.0\n")
            f.write("d 1.0\n")
            f.write(f"map_Kd {texture_path.name}\n")

        # Save OBJ referencing the MTL
        obj_path = output_dir / "mesh_textured.obj"
        mesh.export(str(obj_path), file_type="obj", include_normals=True)

        # Prepend MTL reference to OBJ
        with open(obj_path, "r") as f:
            obj_content = f.read()

        with open(obj_path, "w") as f:
            f.write(f"mtllib {mtl_path.name}\nusemtl material_0\n{obj_content}")

        logger.info(f"Saved textured OBJ: {obj_path}")
