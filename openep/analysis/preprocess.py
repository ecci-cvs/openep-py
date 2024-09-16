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
import numpy as np


class Preprocess:
    """
    Pre-process mapping point and mesh data.
    """
    def __init__(self, case):
        self.case = case

    def find_lat_indices(self, lat_threshold=-10000):
        """
        Extract the indices of the mapping points where the LAT matches provided threshold.

        Returns:
            lat_threshold (ndarray): Indices of mapping points to exclude at threshold value
        """
        electric = self.case.electric
        activation_time = electric.annotations.local_activation_time - electric.annotations.reference_activation_time
        return np.where(activation_time == lat_threshold)[0]
