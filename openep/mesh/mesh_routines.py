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
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    *,
    bcpd_path: Union[str, Path] = "bcpd",
    bcpd_args: Dict[str, Union[str, int, float]],
    work_dir: Optional[Union[str, Path]] = None,
    temp_dir_name: Optional[str] = None,
    log_file: Union[str, Path] = "bcpd.log",
    on_stdout: Optional[Callable[[str], None]] = None,    
    visualise: bool = False,
    keep_files: bool = False,
    strict_flags: bool = False,
) -> Tuple[np.ndarray, Dict[str, Union[str, float]]]:
    """
    Run BCPD and return final registered points + estimated parameters.

    Parameters
    ----------
    source_pts : (M,3) array
        Moving/source point cloud.
    target_pts : (N,3) array
        Fixed/target point cloud.
    bcpd_path : str or Path
        Path to the BCPD executable.
    bcpd_args : dict
        Keyword → value for BCPD flags (e.g. {"beta":2,"lam":10,"outlier":0.1,"s":"Y"}).
    work_dir : str or Path, optional
        Parent directory for a temporary workspace.
    temp_dir_name : str, optional
        If provided, create the workspace as ``Path(work_dir or tempfile.gettempdir())/temp_dir_name``.
        If the directory already exists, a ``FileExistsError`` is raised. If not provided, a
        random directory is created via ``tempfile.mkdtemp(dir=work_dir)`` (previous behavior).        
    log_file : str or Path
        Filename (or absolute path) for logging inside the workspace. Lines from BCPD stdout are
        appended here (with carriage-return progress translated to newlines) when supplied.
    on_stdout : Callable[[str], None], optional
        If provided, called with each logical line of BCPD stdout as it arrives. Useful for GUI live updates.
    visualise : bool
        If True, launch vedo-based interactive viewer.
    keep_files : bool
        If True, do not delete the workspace on exit.
    strict_flags : bool
        If True, error on unknown BCPD flags.

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

    # make workspace
    if temp_dir_name is not None:
        parent = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
        ws = parent / temp_dir_name
        # Avoid accidental reuse of a prior run's files
        ws.mkdir(parents=True, exist_ok=False)
    else:
        ws = Path(tempfile.mkdtemp(dir=work_dir))
    src_txt = ws / "source.txt"
    tgt_txt = ws / "target.txt"
    np.savetxt(src_txt, source_pts, fmt="%.8f")
    np.savetxt(tgt_txt, target_pts, fmt="%.8f")

    # resolve log file path (optional tee)
    log_fp = None
    log_path: Optional[Path] = None
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = ws / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered text file for live tailing
        log_fp = open(log_path, "a", encoding="utf-8")

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
    print("Running: %s", " ".join(shlex.quote(c) for c in cmd))

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

    # # parse trajectory
    # frames: List[np.ndarray] = []
    # for fname in (".optpath.bin", "optpath.bin"):
    #     p = ws / fname
    #     if p.exists():
    #         frames = _read_optpath(p)
    #         break

    # load final registered cloud
    for cand in ("output_y.txt", "y.txt", "Y.txt", "output_x.txt", "x.txt", "X.txt"):
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

    # # visualize if requested
    # if visualise:
    #     _visualise_vedo(source_pts, source_cells, target_pts, target_cells, registered_pts, frames or None)

    # cleanup
    if not keep_files:
        shutil.rmtree(ws, ignore_errors=True)
        print("Removed workspace %s", ws)

    return registered_pts, params