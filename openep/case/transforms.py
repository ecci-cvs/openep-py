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

"""Transformations that can be applied to the point sets owned by a Case."""

import abc

import numpy as np

__all__ = ['Transform', 'MatrixTransform', 'DeformationFieldTransform']


class Transform(abc.ABC):
    """Base class for a transformation that can be applied to an Nx3 array of points."""

    @abc.abstractmethod
    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply this transformation to an array of points.

        Args:
            points (np.ndarray): array of shape Nx3

        Returns:
            np.ndarray: array of shape Nx3 of transformed points
        """


class MatrixTransform(Transform):
    """A rigid, similarity, or affine transformation defined by a 4x4 matrix.

    Args:
        matrix (np.ndarray): 4x4 homogeneous transformation matrix
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float)

    def apply(self, points: np.ndarray) -> np.ndarray:
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]
        return np.dot(np.asarray(points, dtype=float), rotation.T) + translation


class DeformationFieldTransform(Transform):
    """A non-rigid transformation defined by a fitted pycpd deformable registration.

    Args:
        registration: a pycpd `DeformableRegistration` (or `ConstrainedDeformableRegistration`)
            instance that has already been fitted (i.e. `iterate()` has been called until
            convergence). Applying this transform evaluates the fitted registration's
            displacement field at arbitrary query points via `registration.transform_point_cloud`,
            so it does not need to be re-fitted to transform points other than the ones it was
            fitted on.
    """

    def __init__(self, registration):
        self.registration = registration

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.registration.transform_point_cloud(Y=np.asarray(points, dtype=float))
