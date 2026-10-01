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

"""Mesh decimation - reducing a mesh's point count while preserving its overall shape.
"""

import os

import pyvista

__all__ = ['decimate_mesh']


def _decimate_pyvista(mesh: pyvista.PolyData, n_points: int) -> pyvista.PolyData:
    """Decimate via VTK's quadric-based triangle collapse (`PolyData.decimate`)."""

    target_reduction = 1 - (n_points / mesh.n_points)
    return mesh.decimate(target_reduction)


def _decimate_acvd(mesh: pyvista.PolyData, n_points: int) -> pyvista.PolyData:
    """Decimate via uniform remeshing with the ACVD algorithm (PyACVD).

    Tends to produce a more evenly-spaced point distribution than `_decimate_pyvista`,
    at higher computational cost.
    """

    import pyacvd  # optional dependency, only needed for this method

    # try single-threaded execution to avoid oversubscription
    previous_omp_num_threads = os.environ.get('OMP_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        clustering = pyacvd.Clustering(mesh)
        clustering.cluster(n_points)
        return clustering.create_mesh()
    finally:
        if previous_omp_num_threads is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = previous_omp_num_threads


_DECIMATION_METHODS = {
    'pyvista': _decimate_pyvista,
    'acvd': _decimate_acvd,
}


def decimate_mesh(mesh: pyvista.PolyData, n_points: int, method: str = 'pyvista') -> pyvista.PolyData:
    """Decimate a mesh to (approximately) `n_points` points.

    Args:
        mesh (pyvista.PolyData): mesh to decimate.
        n_points (int): target number of points in the decimated mesh. Neither decimation
            method guarantees hitting this exactly - the returned mesh's point count will
            typically be close to, but not exactly, this value.
        method (str): one of:
            - 'pyvista': fast triangle-collapse decimation via `PolyData.decimate`.
            - 'acvd': uniform remeshing via the ACVD algorithm (PyACVD).

    Returns:
        pyvista.PolyData: the decimated mesh. If `n_points` is already at or above the
            mesh's current point count, a copy of `mesh` is returned unchanged.
    """

    if method not in _DECIMATION_METHODS:
        raise ValueError(f"Unknown decimation method: {method!r}. Must be one of {list(_DECIMATION_METHODS)}.")

    if n_points >= mesh.n_points:
        return mesh.copy()

    return _DECIMATION_METHODS[method](mesh, n_points)
