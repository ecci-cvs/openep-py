#!/usr/bin/env python3
"""
Simple manual test for the pre-alignment viewer.

Usage
-----
python tools/test_prealign_viewer.py [--save out.vtp]

Controls inside the viewer
--------------------------
- A : toggle actor edit mode (drag/rotate/scale with mouse)
- R : run coarse RANSAC+ICP (requires open3d)
- Close window to continue/finish
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pyvista as pv

# Import the interactive pre-align hook from the library under test.
# This must exist in mesh_routines as per the feature request.
from mesh_routines import _prealign_interactive_np  # type: ignore


def _ensure_tris(mesh: pv.PolyData) -> pv.PolyData:
    """Return a copy of *mesh* as triangulated PolyData."""
    m = mesh.copy(deep=True)
    if not m.is_all_triangles:
        m = m.triangulate()
    return m


def _mesh_to_arrays(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points, faces) where faces is (F, 3) int array of triangles."""
    m = _ensure_tris(mesh)
    pts = np.asarray(m.points, dtype=np.float64)
    faces = m.faces.reshape(-1, 4)[:, 1:].astype(np.int32, copy=False)
    return pts, faces


def _make_target_sphere(radius: float = 50.0) -> pv.PolyData:
    """Generate a triangulated sphere as the target shell."""
    s = pv.Sphere(
        radius=radius,
        theta_resolution=48,
        phi_resolution=32,
    )
    return _ensure_tris(s)


def _make_transformed_source_from(mesh: pv.PolyData) -> pv.PolyData:
    """Create a misaligned copy of *mesh* via rotation, translation, slight scaling."""
    m = mesh.copy(deep=True)
    # Apply a reproducible transform: rotate, translate, and anisotropic scale
    m.rotate_y(35.0, inplace=True)
    m.translate((30.0, -20.0, 10.0), inplace=True)
    m.scale([1.10, 0.90, 1.00], inplace=True)
    return m


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2)))


def _centroid(pts: np.ndarray) -> np.ndarray:
    return np.mean(pts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive test for _prealign_interactive_np")
    parser.add_argument("--save", type=str, default="", help="Optional path to save adjusted source (vtp/ply).")
    parser.add_argument("--radius", type=float, default=50.0, help="Sphere radius for synthetic meshes.")
    args = parser.parse_args()

    # 1) Build synthetic target & source meshes
    target_mesh = _make_target_sphere(radius=args.radius)
    source_mesh = _make_transformed_source_from(target_mesh)

    # 2) Convert to arrays expected by _prealign_interactive_np
    target_pts, target_faces = _mesh_to_arrays(target_mesh)
    source_pts, source_faces = _mesh_to_arrays(source_mesh)

    # 3) Report simple 'before' metric (centroid distance)
    c_src0 = _centroid(source_pts)
    c_tgt = _centroid(target_pts)
    before_centroid_dist = np.linalg.norm(c_src0 - c_tgt)
    print(f"[before] centroid distance: {before_centroid_dist:.3f} mm")

    # 4) Launch the pre-align viewer (blocks until window closed)
    try:
        adjusted_pts = _prealign_interactive_np(
            source_pts=source_pts,
            target_pts=target_pts,
            source_faces=source_faces,
            target_faces=target_faces,
        )
    except ImportError as e:
        print("ERROR: This test requires 'vedo' (and 'open3d' if you press R).")
        raise

    # 5) Report simple 'after' metrics
    c_src1 = _centroid(adjusted_pts)
    after_centroid_dist = np.linalg.norm(c_src1 - c_tgt)
    shift_rms = _rms(adjusted_pts - source_pts)
    print(f"[after ] centroid distance: {after_centroid_dist:.3f} mm")
    print(f"[delta ] RMS point shift (source → adjusted): {shift_rms:.3f} mm")

    # 6) Visual check: overlay target (gray), adjusted (blue), and original (red wireframe)
    if args.save:
        out = Path(args.save)
        # rebuild a PolyData from adjusted points & original faces
        tri_hdr = np.full((source_faces.shape[0], 1), 3, dtype=np.int32)
        faces_flat = np.hstack([tri_hdr, source_faces]).astype(np.int32).ravel()
        out_mesh = pv.PolyData(adjusted_pts, faces_flat)
        if out.suffix.lower() == ".vtp":
            out_mesh.save(out)
        elif out.suffix.lower() == ".ply":
            out_mesh.save(out)
        else:
            # default to VTP
            out = out.with_suffix(".vtp")
            out_mesh.save(out)
        print(f"[save  ] wrote {out.resolve()}")

    # Build PolyData from adjusted points for visualization
    tri_hdr = np.full((source_faces.shape[0], 1), 3, dtype=np.int32)
    faces_flat = np.hstack([tri_hdr, source_faces]).astype(np.int32).ravel()
    adjusted_mesh = pv.PolyData(adjusted_pts, faces_flat)

    # Launch a PyVista plotter to verify the final alignment
    plotter = pv.Plotter(title="Post-prealign check")
    # Target: gray, slightly transparent
    plotter.add_mesh(target_mesh, color="lightgray", opacity=0.5, smooth_shading=True)
    # Adjusted source: blue
    plotter.add_mesh(adjusted_mesh, color="royalblue", opacity=0.9, smooth_shading=True)
    # Original source: red wireframe overlay
    source_wire = source_mesh.extract_surface().triangulate()
    plotter.add_mesh(
        source_wire,
        color="crimson",
        opacity=1.0,
        style="wireframe",
        line_width=1.0,
    )
    plotter.add_text("Gray: target • Blue: adjusted • Red wireframe: original", font_size=10)
    plotter.show()

if __name__ == "__main__":
    main()