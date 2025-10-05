# OpenEP
# Copyright (c) 2021 OpenEP Collaborators
#
# This file is part of OpenEP.
#
# OpenEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program (LICENSE.txt).  If not, see <http://www.gnu.org/licenses/>

"""
Analyse a mesh - :mod:`openep.mesh.mesh_routines`
=================================================

This module provides methods for calculating the mesh :ref:`surface area and volume
<geometry>`, calculating :ref:`Euclidian and geodesic distances <distances>`
bewteen points on a mesh, and :ref:`identifying the free boundaries
of a mesh <boundaries>`,

.. _geometry:

Calculating the mesh surface area and volume
--------------------------------------------

.. autofunction:: calculate_mesh_volume

.. autofunction:: calculate_field_area

.. autofunction:: point_data_to_cell_data

.. _distances:

Distances between points on a mesh
----------------------------------

.. autofunction:: calculate_vertex_distance

.. autofunction:: calculate_vertex_path


.. _boundaries:

Identifying and analysing the free boundaries of a mesh
-------------------------------------------------------

.. autofunction:: get_free_boundaries

.. autoclass:: FreeBoundary
    :members: separate_boundaries, calculate_lengths, calculate_areas

Calculating mesh properties on a per-region basis
-------------------------------------------------

.. autofunction:: low_field_area_per_region

.. autofunction:: mean_field_per_region

"""

from attr import attrs
from typing import Callable, Dict, Union, Optional, List, Tuple
from pathlib import Path

import numpy as np
import scipy.stats

import pyvista
import pymeshfix
import trimesh
import vedo

__all__ = [
    "get_free_boundaries",
    "calculate_mesh_volume",
    "repair_mesh",
    "point_data_to_cell_data",
    "calculate_field_area",
    "calculate_vertex_distance",
    "calculate_vertex_path",
    "voxelise",
    "low_field_area_per_region",
    "mean_field_per_region",
    "read_bcpd_optpath"
]


def _create_trimesh(pyvista_mesh):
    """Convert a pyvista mesh into a trimesh mesh.

    Args:
        pyvista_mesh (pyvista.PolyData): The pyvista mesh from which the trimesh mesh will be generated

    Returns:
        trimesh_mesh (trimesh.Trimesh): The generated trimesh mesh
    """

    vertices = pyvista_mesh.points
    faces = pyvista_mesh.faces.reshape(pyvista_mesh.n_cells, 4)[:, 1:]  # ignore to number of vertices per face

    return trimesh.Trimesh(vertices, faces, process=False)


class Boundary:
    """Base class for storing information on the boundaries of a mesh.

    Args:
        points (np.ndarray): (N,3) array of coordinates
        lines (np.ndarray): (M, 2) array containing the indices of points connected
            by an edge. Values should be in the interval [0, N-1].
        n_boundaries (int): the number of boundaries
        n_points_per_boundary (np.ndarray): the number of points in each boundary
        original_lines (np.ndarray): (M, 2) array. Same as `lines`, but the indices
            should correspond to those in the original mesh from which boundaries
            were identified.
    """

    def __init__(
        self,
        points: np.ndarray,
        lines: np.ndarray,
        n_boundaries: int,
        n_points_per_boundary: np.ndarray,
        original_lines: np.ndarray
    ):

        self.points = points
        self.lines = lines
        self.n_boundaries = n_boundaries
        self.n_points_per_boundary = n_points_per_boundary
        self.original_lines = original_lines

        if self.n_boundaries == 0:
            self._start_indices = None
            self._stop_indices = None
            return None

        # We'll use the start and stop indices for separating the (N,2) ndarray
        # of indices into separate arrays for each boundary
        start_indices = list(np.cumsum(self.n_points_per_boundary[:-1] - 1))
        start_indices.insert(0, 0)
        self._start_indices = np.asarray(start_indices)

        stop_indices = start_indices[1:]
        stop_indices.append(None)
        self._stop_indices = np.asarray(stop_indices)

    def separate_boundaries(self, original_lines: bool = False):
        """
        Creates a list of numpy arrays where each array contains the indices of
        node pairs in a single boundary.

        Args:
            original_lines (bool):
                If True, `FreeBoundary.original_lines` will be used.
                If False, `FreeBoundary.lines` will be used.

        Returns:
            separate_boundaries (list): a list of numpy arrays - one array per free boundary.
                Each array is of shape Nx2, where N is the number of lines in a given boundary.
                Each array contains the indices of pairs of nodes that make up each line in the
                boundary.

        """

        if self.n_boundaries == 0:
            return np.array([])

        lines = self.original_lines if original_lines else self.lines

        separate_boundaries = [
            lines[start:stop] for start, stop in zip(self._start_indices, self._stop_indices)
        ]

        return separate_boundaries

    def calculate_lengths(self):
        """
        Calculates the length of the perimeter of each free boundary.

        Returns:
            lengths (np.ndarray): the perimeter of each free boundary

        """

        if self.n_boundaries == 0:
            return np.array([])

        lengths = [
            self._line_length(self.points[self.lines[start:stop]]) for
            start, stop in zip(self._start_indices, self._stop_indices)
        ]

        return np.asarray(lengths)

    def _line_length(self, points):
        """
        Calculates the length of a line defined by the positions of a sequence of points.

        Args:
            points (ndarray): Nx3 array of cartesian coordinates of points along the line.

        Returns:
            total_distance (float): length of the line
        """

        distance_between_neighbours = np.sqrt(np.sum(np.square(points[:, 0, :] - points[:, 1, :]), axis=1))
        total_distance = np.sum(distance_between_neighbours)

        return float(total_distance)

class FreeBoundary(Boundary):
    """
    Class for storing information on the free boundaries of a mesh.

    Args:
        points (np.ndarray): (N,3) array of coordinates
        lines (np.ndarray): (M, 2) array containing the indices of points connected
            by an edge. Values should be in the interval [0, N-1].
        n_boundaries (int): the number of free boundaries
        n_points_per_boundary (np.ndarray): the number of points in each boundary
        original_lines (np.ndarray): (M, 2) array. Same as `lines`, but the indices
            should correspond to those in the original mesh from which free boundaries
            were identified.
    """

    def __init__(
        self,
        points: np.ndarray,
        lines: np.ndarray,
        n_boundaries: int,
        n_points_per_boundary: np.ndarray,
        original_lines: np.ndarray
    ):
        super().__init__(
            points,
            lines,
            n_boundaries,
            n_points_per_boundary,
            original_lines,
        )

        self._boundary_meshes = None

    def separate_boundaries(self, original_lines: bool = False):
        """
        Creates a list of numpy arrays where each array contains the indices of
        node pairs in a single boundary.

        Args:
            original_lines (bool):
                If True, `FreeBoundary.original_lines` will be used.
                If False, `FreeBoundary.lines` will be used.

        Returns:
            separate_boundaries (list): a list of numpy arrays - one array per free boundary.
                Each array is of shape Nx2, where N is the number of lines in a given boundary.
                Each array contains the indices of pairs of nodes that make up each line in the
                boundary.

        """
        return super().separate_boundaries(original_lines)

    def calculate_lengths(self):
        """
        Calculates the length of the perimeter of each free boundary.

        Returns:
            lengths (np.ndarray): the perimeter of each free boundary

        """
        return super().calculate_lengths()

    def calculate_areas(self):
        """
        Calculates the cross-sectional area of each boundary.

        Returns:
            areas (np.ndarray): the area of each free boundary

        """

        if self.n_boundaries == 0:
            return np.array([])

        if self._boundary_meshes is None:
            self._create_boundary_meshes()

        areas = [mesh.area for mesh in self._boundary_meshes]

        return np.asarray(areas)

    def _create_boundary_meshes(self):
        """
        Create a pyvista.PolyData mesh for each boundary.

        This determines the geometric centre of the boundary, adds a point at the centre, then
        creates a new mesh in which every edge forms a triangle with the central point. This
        thus creates a surface of the boundary, from which the cross-sectional area can be calculated.
        """

        boundary_meshes = []
        boundaries = self.separate_boundaries(original_lines=False)

        for boundary in boundaries:

            points = self.points[boundary[:, 0]]
            center = np.mean(points, axis=0)
            points = np.vstack([center, points])

            num_points = points.shape[0]
            n_vertices_per_node = np.full(num_points - 1, fill_value=3, dtype=int)

            vertex_one = np.zeros(num_points - 1, dtype=int)  # all triangles include the central point, index 0
            vertex_two = np.arange(1, num_points)
            vertex_three = np.roll(vertex_two, shift=-1)
            faces = np.vstack([n_vertices_per_node, vertex_one, vertex_two, vertex_three]).T.ravel()

            boundary_meshes.append(pyvista.PolyData(points, faces))

        self._boundary_meshes = boundary_meshes

        return None


def get_free_boundaries(mesh):
    """
    Determines the freeboundary/outlines of the 3-D mesh.

    Args:
        mesh (pyvista.PolyData): An open mesh for which the free boundaries will be determined.

    Returns:
        free_boundaries (openep.mesh.FreeBoundary):
            The free boundaries of the open mesh.
    """

    tm_mesh = _create_trimesh(mesh)
    boundaries = tm_mesh.outline().entities

    if boundaries.size == 0:
        return FreeBoundary(
            points=np.array([]),
            lines=np.array([]),
            n_boundaries=0,
            n_points_per_boundary=np.array([]),
            original_lines=np.array([]),
        )

    # determine information about each boundary
    original_indices = np.concatenate([line.points for line in boundaries])
    new_indices = np.arange(original_indices.size)

    n_points_per_boundary = np.asarray([line.points.size for line in boundaries])
    n_boundaries = boundaries.size

    # Create an array pairs of neighbouring nodes for each boundary
    original_lines = np.vstack([original_indices[:-1], original_indices[1:]]).T
    new_lines = np.vstack([new_indices[:-1], new_indices[1:]]).T

    # Ignore the neighbours that are part of different boundaries
    keep_lines = np.full_like(new_lines[:, 0], fill_value=True, dtype=bool)
    keep_lines[n_points_per_boundary[:-1].cumsum()-1] = False

    original_lines = original_lines[keep_lines]
    new_lines = new_lines[keep_lines]

    # Get the {x,y,z} coordinates of the first node in each pair
    points = tm_mesh.vertices[original_indices]

    return FreeBoundary(
        points=points,
        lines=new_lines,
        n_boundaries=n_boundaries,
        n_points_per_boundary=n_points_per_boundary,
        original_lines=original_lines,
    )


def calculate_mesh_volume(
    mesh: pyvista.PolyData,
    fill_holes: bool = True,
) -> float:
    """
    Calculate the volume of a mesh.

    Args:
        mesh (PolyData): mesh for which the volume will be calculated
        fill_holes: if True, holes in the mesh are filled. If holes are present the volume is meaningless unless
        they are filled.

    Returns:
        The volume of the mesh.
    """

    if fill_holes:
        mesh = repair_mesh(mesh)

    return mesh.volume


def repair_mesh(mesh: pyvista.PolyData) -> pyvista.PolyData:
    """
    Fill the holes of a mesh to make it watertight.

    Args:
        mesh (PolyData): mesh to be repaired.

    Returns:
        mesh (PolyData): the repaired mesh.
    """
    mf = pymeshfix.MeshFix(mesh)
    mf.repair()

    return mf.mesh


def point_data_to_cell_data(mesh: pyvista.PolyData, field: np.ndarray) -> np.ndarray:
    """
    Calculate a per-triangle field from the given per-vertex field. For each triangle the mean of the vertex values is
    calculated as the triangle value.

    Args:
        mesh (PolyData): PolyData mesh
        field: per-vertex field to convert

    Returns:
        np.ndarray per-triangle field
    """
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    return field[faces].mean(axis=1)


def calculate_field_area(
    mesh: pyvista.PolyData,
    field: np.ndarray,
    threshold: float,
) -> float:
    """
    Calculate the total surface area of cells whose corresponding values in `field` are
    less than or equal to the given threshold.

    Args:
        mesh (PolyData): pyvista mesh
        field (ndarray): scalar values that will be filtered based on the given threshold
        threshold (float): cells with values in `field` less than or equal to this value
            will be included when calculating the surface area.

    Returns:
        float: total area of selected cells

    Note
    ----
    This function will add the area of each cell to the mesh as mesh.cell_data if it is not already
    present. This is to prevent calculating cell areas every time this function is called.

    Note
    ----
    This function makes use of :func:`openep.mesh.mesh_routines.point_data_to_cell_data`

    """

    if 'Area' not in mesh.cell_data:

        areas = mesh.compute_cell_sizes(
            length=False,
            area=True,
            volume=False,
        )['Area']

        mesh.cell_data.set_array(areas, 'Area')

    tri_field = point_data_to_cell_data(mesh, field)
    selection = tri_field <= threshold
    selected_areas = areas[selection]

    return selected_areas.sum()


def calculate_vertex_distance(
    mesh: pyvista.PolyData,
    start_index: int,
    end_index: int,
    metric: str = "geodesic",
) -> float:
    """
    Calculate the distance from vertex at `start_idx` to `end_idx`.

    Either the Euclidian or geodesic distance can be calculated.

    Args:
        mesh (PolyData): Polydata mesh
        start_index (int) : index of starting vertex
        end_index (int) : index of ending vertex
        metric (str): The distance metric to use. The distance function can
            be 'geodesic' or 'euclidian'.

    Returns:
        float: distance between vertices
    """

    if metric not in {"geodesic", "euclidian"}:
        raise ValueError("metric must be on of: geodesic, euclidian")

    if metric == "euclidian":

        distance = np.linalg.norm(
            mesh.points[start_index] - mesh.points[end_index]
        )

        return distance

    try:
        distance = mesh.geodesic_distance(start_index, end_index)
    except(ValueError):
        distance = np.nan

    return distance


def calculate_vertex_path(
    mesh: pyvista.PolyData,
    start_index: int,
    end_index: int
) -> np.ndarray:
    """
    Calculate the path from vertex at `start_idx` to `end_idx` as a path of vertices through the mesh.

    This is a wrapper around pyvista.PolyData.geodesic, but it returns an empty array if no path
    exist between the two vertices.

    Args:
        mesh (PolyData): Polydata mesh
        start_index (int) : index of starting vertex
        end_index (int) : index of ending vertex

    Returns:
        ndarray: Array of vertex indices defining the path
    """

    try:
        path_mesh = mesh.geodesic(start_vertex=start_index, end_vertex=end_index)
        path = np.asarray(path_mesh.point_data['vtkOriginalPointIds'][path_mesh.lines[1:]])

    except(ValueError):
        path = np.array([])

    return path


def _get_unreferenced_points(mesh):
    """Determine indices of points not referenced in the triangulation"""

    indices = np.arange(mesh.n_points)
    referenced_indices = np.unique(mesh.faces.reshape(mesh.n_faces, 4)[:, 1:].ravel())
    unreferenced_indices = np.isin(indices, referenced_indices, assume_unique=True, invert=True)

    return unreferenced_indices

def _determine_voxel_bins(mesh, edge_length, border=10):
    """Determine the bins to voxelise a mesh.
    
    Args:
        mesh (pyvista.PolyData): Mesh to be voxelised.
        edge_length (float): Edge length of each voxel, in mm.
        border (float): Minimum border around the mesh, in mm. 

    Returns:
        (np.ndarray): Bins that can be used to construct arrays of voxels.

    """

    def _round_up(value, nearest):
        return np.ceil(value / nearest) * nearest

    def _round_down(value, nearest):
        return np.floor(value / nearest) * nearest

    low_values = np.asarray(mesh.bounds[::2])
    high_values = np.asarray(mesh.bounds[1::2])

    low_values = _round_down(low_values, nearest=border) - border / 2
    high_values = _round_up(high_values, nearest=border) + border / 2

    x_low, y_low, z_low = low_values
    x_high, y_high, z_high = high_values

    x_bins = np.arange(x_low, x_high + edge_length, edge_length)
    y_bins = np.arange(y_low, y_high + edge_length, edge_length)
    z_bins = np.arange(z_low, z_high + edge_length, edge_length)

    bin_edges = [x_bins, y_bins, z_bins]
    bin_centres = [
        x_bins[:-1] + edge_length / 2,
        y_bins[:-1] + edge_length / 2,
        z_bins[:-1] + edge_length / 2,
    ]

    return bin_edges, bin_centres


def voxelise(
    mesh: pyvista.PolyData,
    thickness: Union[float, np.ndarray] = 2,
    n_surfaces: int = 11,
    edge_length: float = 1,
    extract_myocardium: bool = False,
) -> pyvista.PolyData:
    """Voxelise a surface mesh.

    Args:
        mesh (PolyData): Surface mesh to be voxelised.
        thickness (float or np.ndarray): If a float, this defines to thickness of the myocardium.
            An array of thicknesses - one per point in the mesh - can be passed to create a voxelised
            mesh with heterogenous thickness.
        edge_length (float): Length of the voxel edges, in mm.
        n_surfaces (int): A series of surface meshes are created by interpolating points between the
            endocardium and epicardium. Points from this series of meshes are used to determine which voxels
            should be filled. The smaller the voxel edge length, the larger :attr:`n_surfaces`
            should be.
        extract_myocardium (bool, optional): If True the voxelised myocardium will be extracted and
            returned. If False, the voxels in a StructuredGrid will be labelled as filled (1) or empty (0),
            and this data stored as point data in the returned mesh.

    Returns:
        StructuredGrid: The voxelised mesh.
    """

    # Don't make any changes to the mesh
    mesh = mesh.copy(deep=True)

    # Remove points not referenced by the triangulation
    not_referenced = _get_unreferenced_points(mesh)
    mesh.remove_points(not_referenced, inplace=True)

    # Compute normals and set thicknesses
    mesh.compute_normals(inplace=True, auto_orient_normals=True, cell_normals=False, point_normals=True)
    thickness = np.full(mesh.n_points, fill_value=thickness, dtype=float) if isinstance(thickness, float) else thickness
    mesh.point_data['Thickness'] = thickness

    # Calculate voxel bins and create output mesh
    bin_edges, bin_centres = _determine_voxel_bins(mesh, edge_length=edge_length)
    bin_centres_x, bin_centres_y, bin_centres_z = bin_centres

    XX, YY, ZZ = np.meshgrid(bin_centres_x, bin_centres_y, bin_centres_z, indexing='ij')  # use bin centres, use matrix index ordering (ij)

    voxels = pyvista.StructuredGrid(XX, YY, ZZ)
    n_voxels_x, n_voxels_y, n_voxels_z = voxels.x.shape
    voxel_filled = np.zeros(voxels.n_points, dtype=int)  # keep track of which voxels are filled. 0: empty, 1: filled

    # Create a series - of open meshes and voxelise each mesh
    for shell_distance in np.linspace(0, 1, n_surfaces):

        shell = mesh.copy(deep=True)
        shell.points += mesh.point_data['Normals'] * mesh.point_data['Thickness'][:, np.newaxis] * shell_distance
        shell.subdivide_adaptive(max_edge_len=edge_length, inplace=True)

        mesh_binned = scipy.stats.binned_statistic_dd(
            sample=np.asarray(shell.points),
            values=np.zeros(shell.n_points),
            statistic='count',
            bins=bin_edges,
            expand_binnumbers=True,
        )

        x_indices, y_indices, z_indices = mesh_binned.binnumber - 1
        bin_indices = np.ravel_multi_index(
            [x_indices, y_indices, z_indices],
            dims=np.asarray([n_voxels_x, n_voxels_y, n_voxels_z], dtype=int),
            order='F',  # pyvista uses Fortran ordering for point data
        )
        bin_indices = np.unique(bin_indices)

        voxel_filled[bin_indices] = 1

    voxels.point_data['Filled'] = voxel_filled

    if extract_myocardium:
        voxels = voxels.extract_points(voxel_filled.astype(bool))

    return voxels


def low_field_area_per_region(
    mesh: pyvista.PolyData,
    field: np.ndarray,
    cell_region: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    A per-region equivalent of :func:`openep.mesh.mesh_routines.calculate_field_area`.

    For each region of the mesh, calculates the total surface area of cells whose corresponding
    values in `field` are less than or equal to the given threshold.

    Regions must be defined by unique integers, one per region.

    Args:
        mesh (PolyData): pyvista mesh
        field (np.ndarray): scalar values that will be filtered based on the given threshold.
            If field corresponds to point data, this will be transformed to cell data.
        cell_region (np.ndarray): region each cell belongs to (size of array should be mesh.n_cells)
        threshold (float): cells with values in `field` less than or equal to this value
            will be included when calculating the surface area.

    Returns:
        np.ndarray: total area of selected cells in each region

    Note
    ----
    This function will add the area of each cell to the mesh as mesh.cell_data if it is not already
    present. This is to prevent calculating cell areas every time this function is called.

    """

    if 'Area' not in mesh.cell_data:

        areas = mesh.compute_cell_sizes(
            length=False,
            area=True,
            volume=False,
        )['Area']

        mesh.cell_data.set_array(areas, 'Area')

    field_association = 'point' if field.size == mesh.n_points else 'cell'
    if field_association == 'point':
        field = point_data_to_cell_data(mesh, field)

    unique_regions = np.unique(cell_region)
    low_field_areas = np.full(unique_regions.size, fill_value=np.nan)
    for index, region in enumerate(unique_regions):

        region_mask = cell_region == region
        field_mask = field[region_mask] <= threshold

        region_areas = mesh.cell_data['Area'][region_mask]
        low_field_areas[index] = np.sum(region_areas[field_mask])

    return low_field_areas


def mean_field_per_region(mesh, field, cell_region):
    """Calculate the mean value of a field for each region of a mesh.

    Regions must be defined by unique integers, one per region.

    Args:
        mesh (PolyData): pyvista mesh
        field (np.ndarray): scalar values that will be averaged per region. If field corresponds to
            point data, this will be transformed to cell data.
        cell_region (np.ndarray): region each cell belongs to (size of array should be mesh.n_cells)

    Returns:
        np.ndarray: average of field in each region

    """

    field_association = 'point' if field.size == mesh.n_points else 'cell'
    if field_association == 'point':
        field = point_data_to_cell_data(mesh, field)

    unique_regions = np.unique(cell_region)
    mean_field_values = np.full(unique_regions.size, fill_value=np.nan)
    for index, region in enumerate(unique_regions):

        region_mask = cell_region == region
        mean_field_values[index] = np.nanmean(field[region_mask])

    return mean_field_values

# --------------------------------------------------------------------------- #
# Pre-alignment helpers (optional, imported only if used)
# --------------------------------------------------------------------------- #
def _prealign_interactive_np(source_mesh, target_mesh, voxel_size) -> np.ndarray:
    """
    Launch an interactive viewer to pre-align *source_mesh* to *target_mesh* (vedo.Mesh).
    - Press **r** to run coarse RANSAC+ICP (Open3D), applied in place to the source.
    - Press **a** to toggle manual/auto status text (no change to VTK default 'a').
    - Close the window to continue; returns possibly modified source points as (M,3) array.
    Parameters
    ----------
    source_mesh : vedo.Mesh
        Moving/source mesh. Modified in-place by manual edits or auto pre-align.
    target_mesh : vedo.Mesh
        Fixed/target mesh (displayed as gray).
    voxel_size : float
        Approximate voxel size of the meshes, in mm. Used to set parameters for
        coarse registration.
    Returns
    -------
    np.ndarray
        Current source vertex positions after the window is closed.
    """
    try:
        import vedo  # type: ignore
    except Exception as e:
        raise ImportError("vedo is required for prealign_interactive") from e

    # Style the provided meshes directly; operate in-place
    src = source_mesh.c("blue").alpha(0.8)
    tgt = target_mesh.c("gray").alpha(0.5)

    plt = vedo.Plotter(size=(900, 600), title="Pre-align: Source (blue) vs Target (gray)")
    banner = vedo.Text2D(f"r: auto re-align  •  a: toggle manual align  •  close to continue\n",
                         pos="top-left", c="black")
    banner2 = vedo.Text2D(f"Coarse-register using:\n"
                           f"FPFH features + RANSAC (global) using voxel size = {voxel_size}mm\n"
                           f"Point-to-plane ICP (refinement)",
                           pos="bottom-right", s=0.5, c="black")
    # status label (bottom-left), updated when toggling manual edit
    status = vedo.Text2D("", pos="bottom-left", c="gray")
    plt.add([tgt, src, banner, banner2, status])


    # state for edit mode
    state = {"editing": False}

    def _update_status():
        if state["editing"]:
            try:
                status.text(
                    "Mode: Manual\n"
                    "• left-drag = rotate (hold Ctrl for screen-plane rotate)\n"
                    "• Shift + left-drag = translate\n"
                    "• right-drag = scale"
                )
                status.color("tomato")  # stands out a bit
            except Exception:
                pass
        else:
            try:
                status.text(
                    "Mode: Auto"
                )
                status.color("gray")
            except Exception:
                pass

    def _on_key(evt):
        k = evt.keypress
        if k == "r":
            vedo.printc("[prealign] running coarse registration …", c="green")
            try:
                _prealign_carto_mri(src, tgt, voxel_size)
                vedo.printc("[prealign] done", c="cyan")
            except Exception as ee:
                vedo.printc(f"[prealign] failed: {ee}", c="red")
            plt.render()
        elif k == "a":
            state["editing"] = not state["editing"]
            _update_status()
            plt.render()
        # any other keys are ignored

    plt.add_callback("keypress", _on_key)
    _update_status()
    plt.show(axes=0, interactive=True)
    return src.points()


def _prealign_carto_mri(source_mesh, target_mesh, voxel_size: Optional[float] = None) -> np.ndarray:
    """
    Coarse-register a sparse 'source' shell to a dense 'target' shell using Open3D:
      - FPFH features + RANSAC (global)
      - Point-to-plane ICP (refinement)
    The *source_mesh* is modified **in place** and the 4x4 transform matrix is returned.
    """
    try:
        import open3d as o3d  # type: ignore
    except Exception as e:
        raise ImportError("open3d is required for prealign_interactive") from e

    # 1) vedo.Mesh -> Open3D point cloud
    def _to_o3d(vedo_mesh) -> "o3d.geometry.PointCloud":
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(vedo_mesh.points(), dtype=float).copy())
        pc.estimate_normals()
        return pc

    src_pc = _to_o3d(source_mesh)
    tgt_pc = _to_o3d(target_mesh)

    # 2) basic preprocess + FPFH
    def _preprocess(pc, voxel_size: Optional[float] = None):
        dpc = pc.voxel_down_sample(voxel_size) if voxel_size else pc
        dpc.estimate_normals()
        radius = (5 * voxel_size) if voxel_size else 0.3
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            dpc, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
        )
        return dpc, fpfh

    src_d, src_f = _preprocess(src_pc, voxel_size)
    tgt_d, tgt_f = _preprocess(tgt_pc, voxel_size)

    # 3) RANSAC global
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f,
        mutual_filter=True,
        max_correspondence_distance=10.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(10.0),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 500),
    )
    T_init = ransac.transformation

    # 4) ICP refine (point-to-plane)
    icp = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d,
        max_correspondence_distance=5.0,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    T_final = icp.transformation

    # 5) apply to vedo source in place (expects 4x4 numpy)
    try:
        source_mesh.apply_transform(T_final)
    except Exception:
        # fallback: manual transform of vertices
        P = np.asarray(source_mesh.points(), dtype=float)
        P_h = np.c_[P, np.ones((P.shape[0], 1))]
        P_t = (P_h @ T_final.T)[:, :3]
        source_mesh.points(P_t)
    return T_final

def read_optpath(path: Path) -> List[np.ndarray]:
    """
    Parse BCPD’s binary trajectory file `optpath.bin` following demo/optpath.m:

    int32 N          # number of target points (unused here)
    int32 D          # spatial dimension (2 or 3)
    int32 M          # number of source points
    int32 L          # number of saved iterations
    double T[D*M*L]  # trajectory of source: Y(:)
    double X[D*N]    # final target cloud (skipped)

    Returns
    -------
    frames : list of (M, D) float64 arrays, one per iteration
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        if header.size < 4:
            raise ValueError(f"{path} is too short for optpath header")
        _, D, M, L = header
        count = int(D) * int(M) * int(L)
        T = np.fromfile(f, dtype=np.float64, count=count)
        # skip the final target cloud: D * N doubles
        # np.fromfile(f, dtype=np.float64, count=D*header[0])
    if T.size != count:
        raise ValueError(f"Unexpected trajectory length in {path}")
    # reshape in Fortran order to match MATLAB's [D x M x L]
    T = T.reshape((D, M, L), order="F")
    return [T[:, :, k].T for k in range(L)]

def bcpd_register(
    source_mesh,
    target_mesh,
    *,
    bcpd_path: Union[str, Path] = "bcpd",
    bcpd_args: Dict[str, Union[str, int, float]],
    work_dir: Optional[Union[str, Path]] = None,
    on_stdout: Optional[Callable[[str], None]] = None,
    keep_files: bool = True,
    strict_flags: bool = False,
    prealign_interactive: bool = False,
) -> Tuple[np.ndarray, Dict[str, Union[str, float]]]:
    """
    Run BCPD and return final registered points + estimated parameters.

    Parameters
    ----------
    source_mesh : vedo.Mesh or pyvista.PolyData
        Moving/source mesh; its points will be written to BCPD input.
    target_mesh : vedo.Mesh or pyvista.PolyData
        Fixed/target mesh; its points will be written to BCPD input.
    bcpd_path : str or Path
        Path to the BCPD executable.
    bcpd_args : dict
        Keyword → value for BCPD flags (e.g. {"beta":2,"lam":10,"outlier":0.1,"s":"Y"}).
    work_dir : str or Path
        Workspace directory. All files (inputs/outputs) are placed here.
        If the directory exists, it will be removed before the run to ensure a clean workspace.
    on_stdout : Callable[[str], None], optional
        If provided, called with each logical line from BCPD stdout (CR- or LF-terminated).
    keep_files : bool
        If True, do not delete the workspace on exit.
    strict_flags : bool
        If True, error on unknown BCPD flags.
    prealign_interactive : bool
        If True, launch a vedo viewer to allow manual/auto pre-alignment before running BCPD.
        Close the window to proceed.
    Returns
    -------
    registered : (M,3) array
        The final deformed source cloud ("y").
    params : dict
        Key → value from `bcpd.param` (if produced).
    """
    import shutil
    import shlex
    import tempfile
    import subprocess

    # if mesh is pyvista.PolyData
    if isinstance(source_mesh, pyvista.PolyData):
        source_mesh = vedo.Mesh(source_mesh)
    if isinstance(target_mesh, pyvista.PolyData):
        target_mesh = vedo.Mesh(target_mesh)

    # Extract raw point clouds from vedo meshes
    source_pts = source_mesh.points()
    target_pts = target_mesh.points()

    _FLAG_ALIASES: Dict[str, str] = {
        "beta": "b",
        "lam": "l",
        "outlier": "w",
        "kappa": "k",
        "gamma": "g",
        "kernel_id": "G",  # 0=Gauss,1=IMQ,2=RatQuad,3=Laplace
        # 's' passed verbatim e.g. "-sY"
    }

    # validate shapes
    for name, arr in (("source_pts", source_pts), ("target_pts", target_pts)):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} must be (N,3)")

    # numeric sanity
    for key in ("beta", "lam", "kappa", "gamma", "outlier"):
        if key in bcpd_args:
            v = float(bcpd_args[key])
            if key == "outlier" and not (0 < v < 1):
                raise ValueError("outlier must be in (0,1)")
            if key != "outlier" and v <= 0:
                raise ValueError(f"{key} must be positive")

    # make workspace: use work_dir directly and clean if it already exists
    if work_dir is None:
        raise ValueError("work_dir must be provided and will be used as the workspace")
    ws = Path(work_dir)
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)

    src_txt = ws / "source.txt"
    tgt_txt = ws / "target.txt"
    np.savetxt(src_txt, source_pts, fmt="%.8f")
    np.savetxt(tgt_txt, target_pts, fmt="%.8f")

    # setup logging: always place inside workspace
    log_basename = "bcpd.log"
    log_path = ws / log_basename
    log_fp = open(log_path, "a", encoding="utf-8")

    # --- optional interactive pre-alignment --------------------------------
    if prealign_interactive:
        try:
            source_pts = _prealign_interactive_np(source_mesh, target_mesh, voxel_size=3.0)
            source_mesh.points(source_pts)
        except ImportError as e:
            log_fp.write(f"[prealign] skipped: {e}\\n")
            log_fp.flush()

    # build command
    cmd = [str(bcpd_path), "-x", str(tgt_txt), "-y", str(src_txt)]

    for key, val in bcpd_args.items():
        flag = _FLAG_ALIASES.get(key, key)
        if strict_flags and len(flag) > 1 and flag not in _FLAG_ALIASES.values():
            raise ValueError(f"Unknown BCPD option '{key}'")
        if flag == "s" and isinstance(val, str):
            cmd.append(f"-s{val}")
        else:
            cmd.extend([f"-{flag}", str(val)])

    print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"Workspace: {ws}")
    
    # execute BCPD
    proc = subprocess.Popen(
        cmd, cwd=ws, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout

    def _iter_records(stream):
        """Yield logical records from a text stream, splitting on CR or LF.
        Many CLIs print progress with carriage returns ("\r"). We treat both
        "\r" and "\n" as record separators so GUIs/logs can show incremental lines.
        """
        buf: List[str] = []
        while True:
            ch = stream.read(1)
            if ch == "" or ch is None:
                break
            if ch in ("\r", "\n"):
                if buf:
                    yield "".join(buf)
                buf.clear()
                continue
            buf.append(ch)
        if buf:
            yield "".join(buf)

    for rec in _iter_records(proc.stdout):
        if on_stdout is not None:
            try:
                on_stdout(rec)
            except Exception:
                # do not break the run if UI callback fails
                pass
        if log_fp is not None:
            log_fp.write(rec + "\n")
            log_fp.flush()
        else:
            print(rec)
    
    proc.wait()
    if log_fp is not None:
        log_fp.close()
    if proc.returncode != 0:
        raise RuntimeError(f"BCPD exited {proc.returncode}; see log")

    # load final registered cloud
    #TODO: clean up file paths
    for cand in ("output_y.txt", "y.txt", "Y.txt"):
        if (ws / cand).exists():
            registered_pts = np.loadtxt(ws / cand)
            break
    else:
        raise FileNotFoundError("No registered output file found in workspace")

    # parse params if present
    params: Dict[str, Union[str, float]] = {}
    pfile = ws / "bcpd.param"
    if pfile.exists():
        for ln in pfile.read_text().splitlines():
            if "=" in ln:
                k, v = map(str.strip, ln.split("=", 1))
                params[k] = float(v) if v.replace(".", "", 1).isdigit() else v

    # cleanup
    if not keep_files:
        shutil.rmtree(ws, ignore_errors=True)
        print("Removed workspace %s", ws)

    return registered_pts, params