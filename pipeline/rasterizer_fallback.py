"""Pure-PyTorch fallback for Hunyuan3D's ``custom_rasterizer_kernel`` C++ extension.

When the CUDA toolkit is not installed (CUDA_HOME is None) the compiled
``custom_rasterizer_kernel`` pybind11 module cannot be built.  This module
provides a *functionally equivalent* pure-Python/PyTorch implementation of the
two functions consumed by the Paint pipeline:

    custom_rasterizer_kernel.rasterize_image(V, F, D, width, height, occ_trunc, use_depth_prior)
    -> (findices, barycentric)

The algorithm is a direct port of the C++ CPU path in
``custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp``.

Performance:  This fallback is ~10--50x slower than the compiled CUDA kernel
because it loops over triangles in Python.  For the Paint pipeline's typical
meshes (~40k faces at 2048x2048 resolution) it takes roughly 30--120 seconds
per rasterize call on a modern CPU, versus <1 second for the compiled version.
This is acceptable for a development/testing workflow but not for production.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

MAXINT = 2147483647


def rasterize_image(V, F, D, width, height, occlusion_truncation, use_depth_prior):
    """Port of ``rasterize_image`` from the C++ custom_rasterizer_kernel.

    Args:
        V: float32 tensor of shape (num_vertices, 4) -- homogeneous clip-space
           positions (x, y, z, w).
        F: int32 tensor of shape (num_faces, 3) -- triangle vertex indices.
        D: float32 tensor -- depth prior (unused when *use_depth_prior* == 0).
        width: int -- image width in pixels.
        height: int -- image height in pixels.
        occlusion_truncation: float -- depth occlusion threshold.
        use_depth_prior: int -- whether to use a depth prior image.

    Returns:
        (findices, barycentric): tuple of tensors.
            findices: int32 (height, width) -- 1-indexed face id per pixel (0 = empty).
            barycentric: float32 (height, width, 3) -- barycentric coords per pixel.
    """
    # Move tensors to CPU/numpy for the rasterisation loop.
    V_np = V.detach().cpu().float().numpy()
    F_np = F.detach().cpu().int().numpy()
    D_np = D.detach().cpu().float().numpy() if (use_depth_prior and D.numel() > 0) else None

    num_faces = F_np.shape[0]

    # Z-buffer stores packed (depth_quantised * MAXINT + face_id) as int64.
    maxint64 = np.int64(MAXINT) * np.int64(MAXINT) + np.int64(MAXINT - 1)
    zbuffer = np.full((height, width), maxint64, dtype=np.int64)

    # Pre-compute screen-space positions for all vertices.
    w_vals = V_np[:, 3]
    # Avoid division by zero
    w_safe = np.where(np.abs(w_vals) < 1e-12, 1e-12, w_vals)
    sx = (V_np[:, 0] / w_safe * 0.5 + 0.5) * (width - 1) + 0.5
    sy = (0.5 + 0.5 * V_np[:, 1] / w_safe) * (height - 1) + 0.5
    sz = V_np[:, 2] / w_safe * 0.49999 + 0.5
    screen = np.stack([sx, sy, sz], axis=1).astype(np.float32)  # (V, 3)

    logger.info(
        "Rasterizer fallback: rasterising %d triangles at %dx%d (this may take a moment)...",
        num_faces, width, height,
    )

    # Rasterise each triangle.
    for fi in range(num_faces):
        i0, i1, i2 = int(F_np[fi, 0]), int(F_np[fi, 1]), int(F_np[fi, 2])
        vt0 = screen[i0]
        vt1 = screen[i1]
        vt2 = screen[i2]

        x_min = int(max(0, min(vt0[0], vt1[0], vt2[0])))
        x_max = int(min(width - 1, max(vt0[0], vt1[0], vt2[0])))
        y_min = int(max(0, min(vt0[1], vt1[1], vt2[1])))
        y_max = int(min(height - 1, max(vt0[1], vt1[1], vt2[1])))

        if x_min > x_max or y_min > y_max:
            continue

        # Vectorised rasterisation over the bounding box of this triangle.
        px_range = np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5
        py_range = np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5
        px_grid, py_grid = np.meshgrid(px_range, py_range)  # (H_bb, W_bb)
        px_flat = px_grid.ravel()
        py_flat = py_grid.ravel()

        # Barycentric coordinates via cross products.
        # signed_area(a, b, c) = (c.x - a.x)*(b.y - a.y) - (b.x - a.x)*(c.y - a.y)
        area = (vt2[0] - vt0[0]) * (vt1[1] - vt0[1]) - (vt1[0] - vt0[0]) * (vt2[1] - vt0[1])
        if abs(area) < 1e-10:
            continue
        inv_area = 1.0 / area

        # beta  = signed_area(a, p, c) / area
        beta = ((vt2[0] - vt0[0]) * (py_flat - vt0[1]) - (py_flat * 0 + vt2[1] - vt0[1]) * (px_flat - vt0[0]))
        # Correct formula: beta_tri = (c.x - a.x)*(p.y - a.y) - (c.y - a.y)*(p.x - a.x)
        beta = ((vt2[0] - vt0[0]) * (py_flat - vt0[1]) - (vt2[1] - vt0[1]) * (px_flat - vt0[0])) * inv_area
        # gamma = signed_area(a, b, p) / area = ((p.x - a.x)*(b.y - a.y) - (b.x - a.x)*(p.y - a.y)) / area
        gamma = ((vt1[0] - vt0[0]) * (py_flat - vt0[1]) - (vt1[1] - vt0[1]) * (px_flat - vt0[0]))
        # Wait -- let me match the C++ formula exactly.
        # C++: beta_tri = calculateSignedArea2(a, p, c) = (c[0]-a[0])*(p[1]-a[1]) - (p[0]-a[0])*(c[1]-a[1])
        beta = ((vt2[0] - vt0[0]) * (py_flat - vt0[1]) - (px_flat - vt0[0]) * (vt2[1] - vt0[1])) * inv_area
        # gamma_tri = calculateSignedArea2(a, b, p) = (p[0]-a[0])*(b[1]-a[1]) - (b[0]-a[0])*(p[1]-a[1])
        gamma = ((px_flat - vt0[0]) * (vt1[1] - vt0[1]) - (vt1[0] - vt0[0]) * (py_flat - vt0[1])) * inv_area
        alpha = 1.0 - beta - gamma

        # In-bounds mask
        valid = (alpha >= 0) & (alpha <= 1) & (beta >= 0) & (beta <= 1) & (gamma >= 0) & (gamma <= 1)

        if not np.any(valid):
            continue

        # Pixel coordinates of valid pixels
        px_valid = (px_flat[valid] - 0.5).astype(np.int32)
        py_valid = (py_flat[valid] - 0.5).astype(np.int32)

        # Depth at each valid pixel
        depth = alpha[valid] * vt0[2] + beta[valid] * vt1[2] + gamma[valid] * vt2[2]

        if use_depth_prior and D_np is not None:
            depth_thres = D_np[py_valid, px_valid] * 0.49999 + 0.5 + occlusion_truncation
            depth_ok = depth >= depth_thres
            px_valid = px_valid[depth_ok]
            py_valid = py_valid[depth_ok]
            depth = depth[depth_ok]

        if len(px_valid) == 0:
            continue

        z_quant = (depth * (2 << 17)).astype(np.int64)
        token = z_quant * np.int64(MAXINT) + np.int64(fi + 1)

        # Update z-buffer (minimum wins)
        for k in range(len(px_valid)):
            px_k = int(px_valid[k])
            py_k = int(py_valid[k])
            if token[k] < zbuffer[py_k, px_k]:
                zbuffer[py_k, px_k] = token[k]

    # Extract face indices and recompute barycentric coordinates from z-buffer.
    findices = np.zeros((height, width), dtype=np.int32)
    barycentric = np.zeros((height, width, 3), dtype=np.float32)

    face_ids = (zbuffer % np.int64(MAXINT)).astype(np.int64)
    valid_mask = face_ids != (MAXINT - 1)

    # Process valid pixels
    valid_y, valid_x = np.where(valid_mask)

    if len(valid_y) > 0:
        f_ids = face_ids[valid_y, valid_x].astype(np.int64)
        findices[valid_y, valid_x] = f_ids.astype(np.int32)

        # Recompute barycentric with perspective correction
        f_zero_indexed = f_ids - 1
        valid_f = f_zero_indexed >= 0
        vy = valid_y[valid_f]
        vx = valid_x[valid_f]
        fi_arr = f_zero_indexed[valid_f].astype(np.int64)

        i0 = F_np[fi_arr, 0]
        i1 = F_np[fi_arr, 1]
        i2 = F_np[fi_arr, 2]

        # Get clip-space vertex positions
        v0 = V_np[i0]  # (N, 4)
        v1 = V_np[i1]
        v2 = V_np[i2]

        # Screen-space positions
        s0 = screen[i0]  # (N, 3)
        s1 = screen[i1]
        s2 = screen[i2]

        # Pixel centres
        px_c = vx.astype(np.float32) + 0.5
        py_c = vy.astype(np.float32) + 0.5

        # Barycentric (same formula as C++ barycentricFromImgcoordCPU)
        area = (s2[:, 0] - s0[:, 0]) * (s1[:, 1] - s0[:, 1]) - (s1[:, 0] - s0[:, 0]) * (s2[:, 1] - s0[:, 1])
        inv_area = np.where(np.abs(area) > 1e-10, 1.0 / area, 0.0)

        beta = ((s2[:, 0] - s0[:, 0]) * (py_c - s0[:, 1]) - (px_c - s0[:, 0]) * (s2[:, 1] - s0[:, 1])) * inv_area
        gamma = ((px_c - s0[:, 0]) * (s1[:, 1] - s0[:, 1]) - (s1[:, 0] - s0[:, 0]) * (py_c - s0[:, 1])) * inv_area
        alpha = 1.0 - beta - gamma

        # Perspective correction: divide by w, then renormalise
        w0 = v0[:, 3]
        w1 = v1[:, 3]
        w2 = v2[:, 3]
        w0_safe = np.where(np.abs(w0) < 1e-12, 1e-12, w0)
        w1_safe = np.where(np.abs(w1) < 1e-12, 1e-12, w1)
        w2_safe = np.where(np.abs(w2) < 1e-12, 1e-12, w2)

        alpha_w = alpha / w0_safe
        beta_w = beta / w1_safe
        gamma_w = gamma / w2_safe
        w_sum = alpha_w + beta_w + gamma_w
        w_sum_safe = np.where(np.abs(w_sum) < 1e-12, 1e-12, w_sum)
        alpha_w /= w_sum_safe
        beta_w /= w_sum_safe
        gamma_w /= w_sum_safe

        barycentric[vy, vx, 0] = alpha_w
        barycentric[vy, vx, 1] = beta_w
        barycentric[vy, vx, 2] = gamma_w

    # Convert to torch tensors on the same device as input
    device = V.device
    findices_t = torch.from_numpy(findices).to(device=device, dtype=torch.int32)
    barycentric_t = torch.from_numpy(barycentric).to(device=device, dtype=torch.float32)

    return findices_t, barycentric_t


def build_hierarchy(*args, **kwargs):
    """Stub -- build_hierarchy is not used by the Paint pipeline."""
    raise NotImplementedError("build_hierarchy is not implemented in the PyTorch fallback")


def build_hierarchy_with_feat(*args, **kwargs):
    """Stub -- build_hierarchy_with_feat is not used by the Paint pipeline."""
    raise NotImplementedError("build_hierarchy_with_feat is not implemented in the PyTorch fallback")
