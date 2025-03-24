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

"""Module containing preprocess classes"""
from openep.mesh.mesh_routines import repair_mesh
import pyvista as pv
import numpy as np
import openep

__all__ = ['Preprocess']

class Preprocess:
    """
    Pre-process mapping point and mesh data.
    """
    def __init__(self, case):
        self._case = case

    def find_lat_indices(self, lat_threshold=-10000):
        """
        Extract the indices of the mapping points where the LAT matches provided threshold.

        Returns:
            lat_threshold (ndarray): Indices of mapping points to exclude at threshold value
        """
        electric = self._case.electric
        activation_time = electric.annotations.local_activation_time - electric.annotations.reference_activation_time
        return np.where(activation_time == lat_threshold)[0]

    def find_mapping_points_at_distance(self, clearance=0.0, from_wall=False):
        """
        Identify the mapping points to exclude based on their distance to a specified clearance.

        This method calculates the distance between mesh points and electrical mapping points.
        If `from_wall` is True, the function will only consider distances from the wall of the mesh.
        Otherwise, it will also take into account points within the mesh volume.

        Args:
            clearance (float, optional): The threshold distance for excluding points.
                Mapping points within this distance from the mesh surface will be excluded. Defaults to 0.0.
            from_wall (bool, optional): If True, only the points close to the mesh wall will be considered for exclusion.
                If False, the method will also exclude points inside the mesh. Defaults to False.

        Returns:
            ndarray: An array of indices of mapping points to be excluded based on the calculated distance and mesh criteria.
        """
        # Add points from wall of mesh (on either side) to exclude
        temp_mesh = self._case.create_mesh()
        distances = openep.case.calculate_distance(
            origin=temp_mesh.points,
            destination=self._case.electric.bipolar_egm.points,
        )
        excludes = (~np.any(distances < clearance, axis=0)).astype(int)

        # Remove points IN the mesh from exclude
        if not from_wall:
            smoothed_mesh = temp_mesh.smooth(n_iter=10, relaxation_factor=0.1, feature_angle=45.0)
            temp_mesh = repair_mesh(smoothed_mesh)
            mapping_points = pv.PolyData(self._case.electric.bipolar_egm.points)
            inside_points = mapping_points.select_enclosed_points(temp_mesh)
            inside_indices = inside_points['SelectedPoints'].view(bool)
            excludes[inside_indices] = 0

        return np.where(excludes)[0]

    def find_mesh_points_at_distance(self, points_of_interest, distance_threshold=5.0):
        """
        Identify mesh points and their corresponding cells that lie within a specified distance from given points.

        This method calculates the distance between a set of input points and the points on the mesh.
        It then identifies mesh cells and boundary points that are within the specified distance threshold.
        This can be used to isolate regions of the mesh that are near certain points of interest.

        Args:
            points_of_interest (ndarray): A numpy array of point coordinates to measure distances from.
            distance_threshold (float, optional): The maximum distance from the input points to consider
                for identifying nearby mesh points and cells.
                All mesh points and cells within this distance will be included. Defaults to 5.0.

        Returns:
            tuple:
                nearby_cell_ids (ndarray): An array of indices representing the mesh cells whose centers are within the
                        distance threshold from the input points.
                bound_point_ids (ndarray): An array of point indices representing the boundary points of the region
                        closest to the input points, within the distance threshold.
        """
        temp_mesh = self._case.create_mesh()

        # Get the points that are within the distance threshold
        distances = openep.case.calculate_distance(points_of_interest, temp_mesh.points)
        nearby_point_ids = np.unique(np.where(distances < distance_threshold)[1])

        # Get the mesh cells whose centers are within the distance threshold
        mesh_cell_centers = temp_mesh.cell_centers().points
        cell_distances = openep.case.calculate_distance(points_of_interest, mesh_cell_centers)
        nearby_cell_ids = np.unique(np.where(cell_distances < distance_threshold)[1])

        # Extract boundaries of the region based on the nearby points
        region_mesh, _ = temp_mesh.remove_points(~np.isin(np.arange(temp_mesh.n_points), nearby_point_ids), mode='all',
                                            inplace=False)
        region_boundaries = openep.mesh.get_free_boundaries(region_mesh)

        if region_boundaries.n_boundaries > 1:
            # If there are multiple boundaries, select the one closest to the selected points
            boundaries_lines = region_boundaries.separate_boundaries()
            boundaries_points = [region_boundaries.points[lines[:, 0]] for lines in boundaries_lines]
            boundaries_com = [np.mean(boundary_points, axis=0) for boundary_points in boundaries_points]
            boundary_index = np.argmin(np.linalg.norm(boundaries_com - np.mean(points_of_interest, axis=0), axis=1))
            bound_point_ids = boundaries_points[boundary_index]
        else:
            bound_point_ids = region_boundaries.points

        # Convert boundary points to mesh point ids
        bound_point_ids = np.argmin(openep.case.calculate_distance(bound_point_ids, temp_mesh.points), axis=1).astype(int)

        return nearby_cell_ids, bound_point_ids