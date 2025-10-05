#!/usr/bin/env python3
"""
Simple manual test for the pre-alignment viewer.

Usage
-----
python tools/test_prealign_viewer.py [--save out.vtp]

Controls inside the viewer
--------------------------
- a : toggle actor edit mode (drag/rotate/scale with mouse)
- r : run coarse RANSAC+ICP (requires open3d)
- Close window to continue/finish
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import vedo

# Import the interactive pre-align hook from the library under test.
# This must exist in mesh_routines as per the feature request.
from openep.mesh.mesh_routines import _prealign_interactive_np  # type: ignore


def _make_target_sphere(radius: float = 50.0) -> "vedo.Mesh":
    """Generate a triangulated sphere as the target shell (vedo)."""
    s = vedo.Sphere(r=radius, res=48, quads=False)
    return s

def _make_transformed_source_from(mesh: "vedo.Mesh") -> "vedo.Mesh":
    """Create a misaligned copy of *mesh* via rotation, translation, slight scaling."""
    m = mesh.clone()
    # Apply a reproducible transform: rotate, translate, and anisotropic scale
    m.rotate_y(35.0)
    # m.pos(m.pos() + (30.0, -20.0, 10.0))
    pts = m.points()
    pts += np.array([30.0, -20.0, 10.0])
    m.points(pts)
    m.scale([1.10, 0.90, 1.00])
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
    source_mesh.c("blue").alpha(0.9)
    target_mesh.c("gray").alpha(0.5)

    # Keep a frozen copy of the original source for later visualization
    original_source = source_mesh.clone()

    # 2) Grab arrays for initial metrics
    target_pts = np.asarray(target_mesh.points(), dtype=float)
    source_pts = np.asarray(source_mesh.points(), dtype=float)

    # 3) Report simple 'before' metric (centroid distance)
    c_src0 = _centroid(source_pts)
    c_tgt = _centroid(target_pts)
    before_centroid_dist = np.linalg.norm(c_src0 - c_tgt)
    print(f"[before] centroid distance: {before_centroid_dist:.3f} mm")

    # 4) Launch the pre-align viewer (blocks until window closed)
    try:
        adjusted_pts = _prealign_interactive_np(source_mesh, target_mesh, voxel_size=3.0)
    except ImportError as e:
        print("ERROR: This test requires 'vedo' (and 'open3d' if you press R).")
        raise

    # 5) Report simple 'after' metrics
    c_src1 = _centroid(adjusted_pts)
    after_centroid_dist = np.linalg.norm(c_src1 - c_tgt)
    shift_rms = _rms(adjusted_pts - source_pts)
    print(f"[after ] centroid distance: {after_centroid_dist:.3f} mm")
    print(f"[delta ] RMS point shift (source → adjusted): {shift_rms:.3f} mm")

    # 6) Optional save of adjusted source (vedo supports .vtp/.ply)
    if args.save:
        out = Path(args.save)
        # source_mesh has been modified in place by the viewer; write it out
        if out.suffix.lower() not in (".vtp", ".ply", ".vtk", ".stl", ".obj"):
            out = out.with_suffix(".vtp")  # default to VTP
        source_mesh.write(str(out))       
        print(f"[save  ] wrote {out.resolve()}")

    # 7) Visual check with vedo: overlay target (gray), adjusted (blue), original (red wireframe)
    check = vedo.Plotter(title="Post-prealign check")
    # Target already gray; add it first
    check.add(tgt := target_mesh.clone().alpha(0.5))
    # Adjusted source (already blue)
    check.add(src_adj := source_mesh.clone().alpha(0.9))
    # Original source as red wireframe
    src_orig = original_source.clone().c("crimson").wireframe(True).lw(1.0)
    check.add(src_orig)
    check.add(vedo.Text2D("Gray: target • Blue: adjusted • Red wireframe: original", pos="top-left"))
    check.show(axes=1, interactive=True)

if __name__ == "__main__":
    main()