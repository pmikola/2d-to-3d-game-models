"""Mesh post-processing between geometry generation and UV unwrapping.

Provides repair, simplification, and validation utilities for generated
3D meshes using trimesh.
"""

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def remove_small_components(mesh, min_face_ratio=0.05):
    """Remove disconnected components smaller than *min_face_ratio* of the
    largest component (by face count).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (may be modified in-place via replacement).
    min_face_ratio : float
        Minimum ratio of face count relative to the largest component.
        Components below this threshold are discarded.

    Returns
    -------
    trimesh.Trimesh
        Mesh with small components removed.
    """
    try:
        components = mesh.split()
    except Exception:
        logger.warning("Failed to split mesh into components; returning as-is.")
        return mesh

    if len(components) <= 1:
        logger.info("Mesh has a single component; nothing to remove.")
        return mesh

    # Sort by face count descending.
    components.sort(key=lambda c: len(c.faces), reverse=True)
    max_faces = len(components[0].faces)
    threshold = max_faces * min_face_ratio

    kept = [c for c in components if len(c.faces) >= threshold]
    removed = len(components) - len(kept)
    logger.info(
        "remove_small_components: kept %d / %d components "
        "(threshold %d faces, ratio %.2f).",
        len(kept),
        len(components),
        int(threshold),
        min_face_ratio,
    )

    if len(kept) == 1:
        return kept[0]

    combined = trimesh.util.concatenate(kept)
    return combined


def fix_topology(mesh):
    """Fix common topological issues on *mesh* in-place.

    Performs the following (when available in the installed trimesh version):
    - Remove degenerate (zero-area) faces
    - Remove duplicate faces
    - Fix face winding / normals
    - Fill holes

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    trimesh.Trimesh
        The same mesh object, repaired in-place.
    """
    # Remove degenerate faces ---------------------------------------------------
    try:
        initial_faces = len(mesh.faces)
        mesh.remove_degenerate_faces()
        removed = initial_faces - len(mesh.faces)
        if removed:
            logger.info("fix_topology: removed %d degenerate faces.", removed)
    except AttributeError:
        logger.warning(
            "fix_topology: mesh.remove_degenerate_faces() not available; skipping."
        )

    # Remove duplicate faces ----------------------------------------------------
    try:
        initial_faces = len(mesh.faces)
        mesh.remove_duplicate_faces()
        removed = initial_faces - len(mesh.faces)
        if removed:
            logger.info("fix_topology: removed %d duplicate faces.", removed)
    except AttributeError:
        logger.warning(
            "fix_topology: mesh.remove_duplicate_faces() not available; skipping."
        )

    # Fix normals ---------------------------------------------------------------
    try:
        trimesh.repair.fix_normals(mesh)
        logger.info("fix_topology: normals fixed.")
    except (AttributeError, Exception) as exc:
        logger.warning("fix_topology: fix_normals failed (%s); skipping.", exc)

    # Fix winding ---------------------------------------------------------------
    try:
        trimesh.repair.fix_winding(mesh)
        logger.info("fix_topology: winding fixed.")
    except (AttributeError, Exception) as exc:
        logger.warning("fix_topology: fix_winding failed (%s); skipping.", exc)

    # Fill holes ----------------------------------------------------------------
    try:
        trimesh.repair.fill_holes(mesh)
        logger.info("fix_topology: holes filled.")
    except (AttributeError, Exception) as exc:
        logger.warning("fix_topology: fill_holes failed (%s); skipping.", exc)

    return mesh


def smooth_mesh(mesh, iterations=5, lamb=0.5):
    """Apply Taubin smoothing to *mesh* in-place.

    Taubin smoothing alternates between positive and negative Laplacian
    steps, preventing the shrinkage typical of plain Laplacian smoothing.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    iterations : int
        Number of smoothing iterations.
    lamb : float
        Positive smoothing factor (the negative factor *nu* is fixed at
        -0.53, a common default).

    Returns
    -------
    trimesh.Trimesh
        The smoothed mesh.
    """
    logger.info(
        "smooth_mesh: applying Taubin smoothing (iterations=%d, lamb=%.3f).",
        iterations,
        lamb,
    )
    try:
        trimesh.smoothing.filter_taubin(
            mesh, lamb=lamb, nu=-0.53, iterations=iterations
        )
    except AttributeError:
        logger.warning(
            "smooth_mesh: trimesh.smoothing.filter_taubin not available; "
            "skipping smoothing."
        )
    except Exception as exc:
        logger.warning("smooth_mesh: Taubin smoothing failed (%s); skipping.", exc)

    return mesh


def decimate_mesh(mesh, target_face_count=None, target_ratio=0.5):
    """Reduce polygon count using quadric error metrics.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    target_face_count : int or None
        Absolute target face count.  Takes priority over *target_ratio*.
    target_ratio : float
        Fraction of current faces to keep (used when *target_face_count* is
        ``None``).

    Returns
    -------
    trimesh.Trimesh
        A new (decimated) mesh.
    """
    current_faces = len(mesh.faces)

    if target_face_count is not None:
        target = int(target_face_count)
    else:
        target = max(4, int(current_faces * target_ratio))

    if target >= current_faces:
        logger.info(
            "decimate_mesh: target (%d) >= current faces (%d); skipping.",
            target,
            current_faces,
        )
        return mesh

    logger.info(
        "decimate_mesh: simplifying from %d to %d faces (ratio %.2f).",
        current_faces,
        target,
        target / current_faces,
    )

    try:
        decimated = mesh.simplify_quadric_decimation(face_count=target)
        logger.info(
            "decimate_mesh: result has %d faces.", len(decimated.faces)
        )
        return decimated
    except AttributeError:
        logger.warning(
            "decimate_mesh: simplify_quadric_decimation not available; "
            "returning original mesh."
        )
        return mesh
    except Exception as exc:
        logger.warning(
            "decimate_mesh: decimation failed (%s); returning original mesh.",
            exc,
        )
        return mesh


def validate_mesh(mesh):
    """Return a dict of quality metrics for *mesh* and log warnings for
    any detected issues.

    Metrics
    -------
    - ``is_watertight``: bool
    - ``is_manifold``: bool (consistent winding)
    - ``num_vertices``: int
    - ``num_faces``: int
    - ``num_degenerate_faces``: int
    - ``num_components``: int
    - ``bounding_box_aspect_ratio``: float (max extent / min nonzero extent)

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    dict
    """
    metrics = {}

    metrics["num_vertices"] = len(mesh.vertices)
    metrics["num_faces"] = len(mesh.faces)

    # Watertight ----------------------------------------------------------------
    try:
        metrics["is_watertight"] = bool(mesh.is_watertight)
    except Exception:
        metrics["is_watertight"] = False

    # Manifold (consistent winding) ---------------------------------------------
    try:
        metrics["is_manifold"] = bool(mesh.is_winding_consistent)
    except Exception:
        metrics["is_manifold"] = False

    # Degenerate faces ----------------------------------------------------------
    try:
        degen_mask = mesh.area_faces == 0.0
        metrics["num_degenerate_faces"] = int(np.sum(degen_mask))
    except Exception:
        metrics["num_degenerate_faces"] = 0

    # Components ----------------------------------------------------------------
    try:
        components = mesh.split()
        metrics["num_components"] = len(components)
    except Exception:
        metrics["num_components"] = 1

    # Bounding box aspect ratio -------------------------------------------------
    try:
        extents = mesh.bounding_box.extents
        nonzero = extents[extents > 0]
        if len(nonzero) > 0:
            metrics["bounding_box_aspect_ratio"] = float(
                nonzero.max() / nonzero.min()
            )
        else:
            metrics["bounding_box_aspect_ratio"] = 0.0
    except Exception:
        metrics["bounding_box_aspect_ratio"] = 0.0

    # Log warnings for detected issues ------------------------------------------
    if not metrics["is_watertight"]:
        logger.warning("validate_mesh: mesh is NOT watertight.")
    if not metrics["is_manifold"]:
        logger.warning("validate_mesh: mesh has inconsistent winding (non-manifold).")
    if metrics["num_degenerate_faces"] > 0:
        logger.warning(
            "validate_mesh: %d degenerate (zero-area) faces detected.",
            metrics["num_degenerate_faces"],
        )
    if metrics["num_components"] > 1:
        logger.warning(
            "validate_mesh: mesh has %d disconnected components.",
            metrics["num_components"],
        )

    logger.info("validate_mesh: %s", metrics)
    return metrics


def remove_slivers(mesh, min_area_ratio=0.001):
    """Remove sliver triangles (near-zero area relative to median).

    Marching cubes often creates very thin triangles at grid boundaries.
    These cause visual spikes and rendering artifacts.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    min_area_ratio : float
        Minimum area as fraction of median face area.
    """
    try:
        areas = mesh.area_faces
        positive = areas[areas > 0]
        if len(positive) == 0:
            return mesh

        median_area = np.median(positive)
        threshold = median_area * min_area_ratio
        good_faces = areas >= threshold
        removed = (~good_faces).sum()

        if removed > 0:
            mesh.update_faces(good_faces)
            mesh.remove_unreferenced_vertices()
            logger.info("remove_slivers: removed %d sliver faces (threshold %.6f).",
                        removed, threshold)
    except Exception as exc:
        logger.warning("remove_slivers: failed (%s); skipping.", exc)

    return mesh


def make_watertight(mesh):
    """Enforce watertight mesh by merging close vertices and filling holes.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    """
    try:
        mesh.merge_vertices()
        logger.info("make_watertight: merged close vertices.")
    except Exception as exc:
        logger.warning("make_watertight: merge_vertices failed (%s).", exc)

    try:
        trimesh.repair.fill_holes(mesh)
        logger.info("make_watertight: filled holes.")
    except Exception as exc:
        logger.warning("make_watertight: fill_holes failed (%s).", exc)

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    return mesh


def repair_and_prepare(mesh, decimate_ratio=None, smooth_iterations=3):
    """High-level repair pipeline: validate, clean, and optionally simplify.

    This is the main entry point intended to be called from the orchestrator
    between geometry generation and UV unwrapping.

    Steps
    -----
    1. Validate the incoming mesh.
    2. Remove small disconnected components.
    3. Fix topology (degenerate faces, duplicates, normals, holes).
    4. Optionally smooth (if *smooth_iterations* > 0).
    5. Optionally decimate (if *decimate_ratio* is given).
    6. Validate the repaired mesh and return it.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh produced by the geometry stage.
    decimate_ratio : float or None
        If provided, simplify to this fraction of current face count.
    smooth_iterations : int
        Number of Taubin smoothing iterations.  Set to 0 to skip.

    Returns
    -------
    trimesh.Trimesh
        The repaired and prepared mesh.
    """
    logger.info("repair_and_prepare: starting mesh repair pipeline.")

    # Step 1 -- initial validation
    logger.info("repair_and_prepare: initial validation.")
    validate_mesh(mesh)

    # Step 2 -- remove slivers (near-zero area faces from marching cubes)
    mesh = remove_slivers(mesh)

    # Step 3 -- remove small components
    mesh = remove_small_components(mesh)

    # Step 4 -- fix topology
    mesh = fix_topology(mesh)

    # Step 5 -- enforce watertight
    mesh = make_watertight(mesh)

    # Step 6 -- optional smoothing
    if smooth_iterations and smooth_iterations > 0:
        mesh = smooth_mesh(mesh, iterations=smooth_iterations)
    else:
        logger.info("repair_and_prepare: smoothing skipped (iterations=0).")

    # Step 7 -- optional decimation
    if decimate_ratio is not None:
        mesh = decimate_mesh(mesh, target_ratio=decimate_ratio)
    else:
        logger.info("repair_and_prepare: decimation skipped (no ratio given).")

    # Step 8 -- recalculate normals after all modifications
    try:
        trimesh.repair.fix_normals(mesh)
        logger.info("repair_and_prepare: normals recalculated.")
    except Exception:
        pass

    # Step 6 -- final validation
    logger.info("repair_and_prepare: final validation.")
    metrics = validate_mesh(mesh)

    logger.info(
        "repair_and_prepare: done. Final mesh: %d verts, %d faces.",
        metrics["num_vertices"],
        metrics["num_faces"],
    )
    return mesh
