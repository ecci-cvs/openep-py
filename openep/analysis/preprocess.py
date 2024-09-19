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
            temp_mesh = repair_mesh(temp_mesh)
            mapping_points = pv.PolyData(self._case.electric.bipolar_egm.points)
            inside_points = mapping_points.select_enclosed_points(temp_mesh)
            inside_indices = inside_points['SelectedPoints'].view(bool)
            excludes[inside_indices] = 0

        return np.where(excludes)[0]

