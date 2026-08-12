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

"""Maths for registering the mesh of one Case onto another.

Currently implements Coherent Point Drift (CPD) registration via pycpd. Structured so
that other registration methods can live alongside it here in future.
"""

import numpy as np
import open3d as o3d
import pycpd
import pyvista
from vtkmodules.vtkCommonTransforms import vtkLandmarkTransform

from .case_routines import calculate_distance
from .transforms import Transform, MatrixTransform, DeformationFieldTransform
from ..mesh.decimation import decimate_mesh

__all__ = ['CPDRegistration', 'LandmarkRegistration', 'ICPRegistration']


_CPD_REGISTRATION_CLASSES = {
    'rigid': pycpd.RigidRegistration,
    'similarity': pycpd.RigidRegistration,
    'affine': pycpd.AffineRegistration,
    'deformable': pycpd.DeformableRegistration,
}

# pycpd.RigidRegistration fits a similarity transform (rotation + translation + isotropic
# scale) by default (scale=True) - 'rigid' forces scale off so it's a true rigid-body fit.
_CPD_RIGID_REGISTRATION_SCALE = {
    'rigid': False,
    'similarity': True,
}


def _rigid_registration_to_matrix(registration):
    """Build a 4x4 matrix from a fitted pycpd RigidRegistration.

    pycpd applies its transform as `TY = s * Y @ R + t` (row-vector convention), while
    MatrixTransform applies `points @ rotation.T + translation` - so `rotation = s * R.T`.
    """

    s, R, t = registration.get_registration_parameters()
    matrix = np.eye(4)
    matrix[:3, :3] = s * R.T
    matrix[:3, 3] = np.ravel(t)
    return matrix


def _affine_registration_to_matrix(registration):
    """Build a 4x4 matrix from a fitted pycpd AffineRegistration.

    pycpd applies its transform as `TY = Y @ B + t` (row-vector convention), while
    MatrixTransform applies `points @ rotation.T + translation` - so `rotation = B.T`.
    """

    B, t = registration.get_registration_parameters()
    matrix = np.eye(4)
    matrix[:3, :3] = B.T
    matrix[:3, 3] = np.ravel(t)
    return matrix


def _nearest_point_indices(points, candidates):
    """For each row in `points`, the index of its nearest row in `candidates`."""

    distance = calculate_distance(origin=points, destination=candidates)
    return np.argmin(distance, axis=1).astype(int)


def _remap_constraint_indices(source_landmarks, target_landmarks, fit_source_points, fit_target_points):
    """Remap correspondence constraint points to indices into decimated fit point arrays.
    """

    source_id = _nearest_point_indices(source_landmarks, fit_source_points)
    target_id = _nearest_point_indices(target_landmarks, fit_target_points)

    source_distance = np.linalg.norm(fit_source_points[source_id] - source_landmarks, axis=1)
    target_distance = np.linalg.norm(fit_target_points[target_id] - target_landmarks, axis=1)
    combined_distance = source_distance + target_distance

    keep = []
    seen_source, seen_target = set(), set()
    for i in np.argsort(combined_distance):
        if source_id[i] in seen_source or target_id[i] in seen_target:
            continue
        seen_source.add(source_id[i])
        seen_target.add(target_id[i])
        keep.append(i)
    keep = np.sort(keep)

    return source_id[keep], target_id[keep]


class CPDRegistration:
    """Coherent Point Drift registration between two point clouds, using pycpd.

    Call `run()` to perform the registration; it returns a `Transform` (see :mod:`openep.case.transforms`)
    mapping source points onto target points - a `MatrixTransform` for rigid/similarity/affine
    registration, or a `DeformationFieldTransform` for deformable registration.

    Args:
        source_points (np.ndarray): Nx3 array of points to be registered onto `target_points`.
        target_points (np.ndarray): Mx3 array of points to register `source_points` onto.
        method (str): one of 'rigid' (translation and rotation only), 'similarity'
            (translation, rotation, and isotropic scaling - pycpd.RigidRegistration's
            default), 'affine' (translation, rotation, and non-isotropic scaling), or
            'deformable'.
        n_iterations (int): number of CPD iterations to run.
        progress_callback (callable, optional): called after each iteration as
            `progress_callback(iteration, source_points)`, where `source_points` is pycpd's
            current estimate of the transformed source points (`registration.TY`).
        source_mesh (pyvista.PolyData, optional): mesh whose points are `source_points` -
            required if `decimate=True` (decimation needs mesh topology, not just points).
        target_mesh (pyvista.PolyData, optional): mesh whose points are `target_points` -
            required if `decimate=True`.
        decimate (bool): if True, fit on decimated copies of `source_mesh`/`target_mesh`
            (see :func:`openep.mesh.decimation.decimate_mesh`) instead of the full-resolution
            `source_points`/`target_points`, for a faster fit on dense meshes.
        decimate_n_points (int): target point count for the decimated working copies.
        decimate_method (str): decimation method to use - see `decimate_mesh`.
        **kwargs: passed through to the underlying pycpd registration class (e.g. `alpha`,
            `beta`, `w`). If `method='deformable'` and both `source_id` and `target_id` are
            given, `pycpd.ConstrainedDeformableRegistration` is used instead of
            `pycpd.DeformableRegistration`. These are indices into `source_points`/
            `target_points`; if `decimate=True`, they are remapped to the nearest points in
            the decimated working copies before fitting (see `_remap_constraint_indices`).
    """

    def __init__(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        method: str = 'rigid',
        n_iterations: int = 50,
        progress_callback=None,
        source_mesh: pyvista.PolyData = None,
        target_mesh: pyvista.PolyData = None,
        decimate: bool = False,
        decimate_n_points: int = 1000,
        decimate_method: str = 'pyvista',
        **kwargs,
    ):
        if method not in _CPD_REGISTRATION_CLASSES:
            raise ValueError(
                f"Unknown CPD registration method: {method!r}. Must be one of {list(_CPD_REGISTRATION_CLASSES)}."
            )

        self.source_points = np.asarray(source_points, dtype=float)
        self.target_points = np.asarray(target_points, dtype=float)
        self.method = method
        self.n_iterations = n_iterations
        self.progress_callback = progress_callback
        self.kwargs = kwargs

        fit_source_points = self.source_points
        fit_target_points = self.target_points

        if decimate:
            if source_mesh is None or target_mesh is None:
                raise ValueError("decimate=True requires both source_mesh and target_mesh.")
            fit_source_points = np.asarray(decimate_mesh(source_mesh, decimate_n_points, method=decimate_method).points)
            fit_target_points = np.asarray(decimate_mesh(target_mesh, decimate_n_points, method=decimate_method).points)

            if 'source_id' in kwargs and 'target_id' in kwargs:
                kwargs = dict(kwargs)
                kwargs['source_id'], kwargs['target_id'] = _remap_constraint_indices(
                    self.source_points[kwargs['source_id']],
                    self.target_points[kwargs['target_id']],
                    fit_source_points,
                    fit_target_points,
                )

        # exposed so callers can see what the fit is actually running on
        self.n_fit_source_points = fit_source_points.shape[0]
        self.n_fit_target_points = fit_target_points.shape[0]

        registration_class = _CPD_REGISTRATION_CLASSES[method]
        if method == 'deformable' and 'source_id' in kwargs and 'target_id' in kwargs:
            registration_class = pycpd.ConstrainedDeformableRegistration
        if method in _CPD_RIGID_REGISTRATION_SCALE:
            kwargs = dict(kwargs, scale=_CPD_RIGID_REGISTRATION_SCALE[method])

        self._registration = registration_class(
            X=fit_target_points,
            Y=fit_source_points,
            **kwargs,
        )

    def run(self, should_stop=None) -> Transform:
        """Run the CPD iteration loop and return the fitted Transform.

        Args:
            should_stop (callable, optional): checked before each iteration; if it
                returns True, iteration stops early and the Transform fitted so far
                (from however many iterations completed) is returned.
        """

        for iteration in range(1, self.n_iterations + 1):
            if should_stop is not None and should_stop():
                break
            self._registration.iterate()
            if self.progress_callback is not None:
                self.progress_callback(iteration, self._registration.TY)

        return self._build_transform()

    def _build_transform(self) -> Transform:

        if self.method in ('rigid', 'similarity'):
            return MatrixTransform(_rigid_registration_to_matrix(self._registration))

        if self.method == 'affine':
            return MatrixTransform(_affine_registration_to_matrix(self._registration))

        return DeformationFieldTransform(self._registration)


_LANDMARK_TRANSFORM_MODES = {
    'rigid': 'SetModeToRigidBody',
    'similarity': 'SetModeToSimilarity',
    'affine': 'SetModeToAffine',
}


class LandmarkRegistration:
    """Landmark-pair registration between two point clouds, using VTK's `vtkLandmarkTransform`.

    Finds the transformation that best aligns `source_points` onto `target_points` in the
    least-squares sense, given known point-to-point correspondence (`source_points[i]`
    corresponds to `target_points[i]`). Call `run()` to perform the registration; it returns
    a `MatrixTransform`.

    Args:
        source_points (np.ndarray): Nx3 array of points to be registered onto `target_points`.
        target_points (np.ndarray): Nx3 array of points to register `source_points` onto -
            `target_points[i]` must correspond to `source_points[i]`.
        method (str): one of 'rigid' (translation and rotation only), 'similarity'
            (translation, rotation, and isotropic scaling), or 'affine' (translation,
            rotation, and non-isotropic scaling).
    """

    def __init__(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        method: str = 'similarity',
    ):
        if method not in _LANDMARK_TRANSFORM_MODES:
            raise ValueError(
                f"Unknown landmark registration method: {method!r}. Must be one of {list(_LANDMARK_TRANSFORM_MODES)}."
            )

        self.source_points = np.asarray(source_points, dtype=float)
        self.target_points = np.asarray(target_points, dtype=float)
        self.method = method

    def run(self) -> Transform:
        """Fit the landmark transform and return the fitted Transform."""

        transform = vtkLandmarkTransform()
        transform.SetSourceLandmarks(pyvista.PolyData(self.source_points).GetPoints())
        transform.SetTargetLandmarks(pyvista.PolyData(self.target_points).GetPoints())
        getattr(transform, _LANDMARK_TRANSFORM_MODES[self.method])()
        transform.Update()

        vtk_matrix = transform.GetMatrix()
        matrix = np.array([[vtk_matrix.GetElement(row, column) for column in range(4)] for row in range(4)])

        return MatrixTransform(matrix)


def _default_max_correspondence_distance(points: np.ndarray) -> float:
    """A scale-aware default `max_correspondence_distance` for ICP,
    10% of the point cloud's bounding-box diagonal.
    """

    extent = points.max(axis=0) - points.min(axis=0)
    return 0.1 * np.linalg.norm(extent)


_ICP_WITH_SCALING = {
    'rigid': False,
    'similarity': True,
}


class ICPRegistration:
    """Iterative Closest Point registration between two point clouds, using Open3D
    (`open3d.pipelines.registration.registration_icp` with point-to-point estimation).

    Args:
        source_points (np.ndarray): Nx3 array of points to be registered onto `target_points`.
        target_points (np.ndarray): Mx3 array of points to register `source_points` onto.
        method (str): one of 'rigid' (translation and rotation only) or 'similarity'
            (translation, rotation, and isotropic scaling) - maps to Open3D's
            `TransformationEstimationPointToPoint(with_scaling=...)`.
        max_correspondence_distance (float, optional): maximum distance between a source/
            target point pair for it to be treated as a correspondence at each iteration.
            Defaults to `_default_max_correspondence_distance(target_points)` if not given.
        max_iterations (int): maximum number of ICP iterations.
        relative_fitness (float): convergence threshold on the relative change in fitness
            (fraction of points with a correspondence) between iterations.
        relative_rmse (float): convergence threshold on the relative change in inlier RMSE
            between iterations.
        init (np.ndarray, optional): initial 4x4 transform guess. Defaults to matching
            `source_points`'s centroid onto `target_points`'s (translation only, no rotation/
            scale) - usually improves convergence when the initial poses are far apart.
    """

    def __init__(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        method: str = 'rigid',
        max_correspondence_distance: float = None,
        max_iterations: int = 100,
        relative_fitness: float = 1e-6,
        relative_rmse: float = 1e-6,
        init: np.ndarray = None,
    ):
        if method not in _ICP_WITH_SCALING:
            raise ValueError(
                f"Unknown ICP registration method: {method!r}. Must be one of {list(_ICP_WITH_SCALING)}."
            )

        self.source_points = np.asarray(source_points, dtype=float)
        self.target_points = np.asarray(target_points, dtype=float)
        self.method = method
        self.max_iterations = max_iterations
        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse

        if max_correspondence_distance is None:
            max_correspondence_distance = _default_max_correspondence_distance(self.target_points)
        self.max_correspondence_distance = max_correspondence_distance

        if init is None:
            init = np.eye(4)
            init[:3, 3] = self.target_points.mean(axis=0) - self.source_points.mean(axis=0)
        self.init = np.asarray(init, dtype=float)

    def run(self) -> Transform:
        """Fit the ICP transform and return the fitted Transform."""

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(self.source_points)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(self.target_points)

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.max_correspondence_distance,
            self.init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                with_scaling=_ICP_WITH_SCALING[self.method],
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.relative_fitness,
                relative_rmse=self.relative_rmse,
                max_iteration=self.max_iterations,
            ),
        )

        matrix = np.asarray(result.transformation)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("ICP produced a non-finite transform matrix.")

        return MatrixTransform(matrix)
