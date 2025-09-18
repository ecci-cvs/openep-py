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

"""Module containing classes for storing ablation data."""

from openep.data_structures.electric import LandmarkPoints
from attr import attrs
import numpy as np

__all__ = ['Ablation', 'AblationForce', 'AblationAutoIndex']


@attrs(auto_attribs=True, auto_detect=True)
class AblationForce:
    """Class for storing data on ablation force.

    Args:
        times (np.ndarray): array of shape N
        force (np.ndarray): array of shape N
        axial_angle (np.ndarray): array of shape N
        lateral_angle (np.ndarray): array of shape N
        points (np.ndarray): array of shape Nx3

    """
    times: np.ndarray = None
    force: np.ndarray = None
    axial_angle: np.ndarray = None
    lateral_angle: np.ndarray = None
    points: np.ndarray = None

    def __repr__(self):
        return f"Ablation forces with {len(self.times)} sites."

    def copy(self):
        """Create a deep copy of AblationForce"""

        ablation_force = AblationForce(
            times=np.array(self.times) if self.times is not None else None,
            force=np.array(self.force) if self.force is not None else None,
            axial_angle=np.array(self.axial_angle) if self.axial_angle is not None else None,
            lateral_angle=np.array(self.lateral_angle) if self.lateral_angle is not None else None,
            points=np.array(self.points) if self.points is not None else None,
        )

        return ablation_force

class AblationAutoIndex:
    """Class for storing automatically collected ablation data.

    NOTE:
        This data is equivalent to .rfindex struct in the openep.mat files.
        It stores Carto3 Visitag automated  ablation tag data.

    Args:
        times (np.ndarray): array of shape N,
            Time stamp of each tag

        average_force (np.ndarray): array of shape N,
            Average force applied at each tag

        max_temp (np.ndarray): array of shape N,
            Maximum recorded catheter tip temperature at each tag

        max_power (np.ndarray): array of shape N,
            Maximum applied radiofrequency power at each tag

        force_time_integral (np.ndarray): array of shape N,
            Force time integral calculated at each tag

    Attribute:
        ablation_points (LandmarkPoints): has .points attribute with array of shape Nx3,
            LandmarkPoints containing Cartesian co-ordinates of each tag
    """

    def __init__(
            self,
            times: np.ndarray = None,
            average_force: np.ndarray = None,
            max_temp: np.ndarray = None,
            max_power: np.ndarray = None,
            force_time_integral: np.ndarray = None,
            points: np.ndarray = None,
    ):
        self.times = times
        self.average_force = average_force
        self.max_temp = max_temp
        self.max_power = max_power
        self.force_time_integral = force_time_integral

        size = points.shape[0] if points is not None else 0
        self.is_ablation = np.ones(size, dtype=bool)
        self.internal_names = np.full(size, "Ablation site 0")
        self.names = np.full(size, "Auto index")

        self.ablation_points = LandmarkPoints(
            points=points,
            is_landmark=self.is_ablation,
            internal_names=self.internal_names,
            names=self.names,
        )

    def __repr__(self):
        return f"Ablation Auto Index (rfindex) with {len(self.times)} sites."

    def add_ablation_site(self, points, names: str, internal_names: list):
        """Add new ablation site(s) as Landmark Point"""
        new_points = np.array(points)
        additional_size = points.shape[0]
        new_internal_names = np.array(internal_names)
        new_names = np.full(additional_size, names)

        if self.ablation_points.n_points:
            new_points = np.append(self.ablation_points.points, points, axis=0)
            new_internal_names = np.append(self.ablation_points.internal_names, new_internal_names, axis=0)
            new_names = np.append(self.ablation_points.names, new_names, axis=0)

        self.is_ablation = np.ones(new_points.shape[0], dtype=bool) if new_points is not None else None
        self.ablation_points = LandmarkPoints(
            points=new_points,
            is_landmark=self.is_ablation,
            internal_names=new_internal_names,
            names=new_names
        )

    def copy(self):
        """Create a deep copy of AblationAutoIndex"""
        ablation_auto_index = AblationAutoIndex(
            times=np.array(self.times) if self.times is not None else None,
            average_force=np.array(self.average_force) if self.average_force is not None else None,
            max_temp=np.array(self.max_temp) if self.max_temp is not None else None,
            max_power=np.array(self.max_power) if self.max_power is not None else None,
            force_time_integral=np.array(self.force_time_integral) if self.force_time_integral is not None else None,
            points=self.ablation_points.points if self.ablation_points is not None else None,
        )
        return ablation_auto_index


@attrs(auto_attribs=True, auto_detect=True)
class Ablation:
    """
    Class for storing ablation data.

    Args:
        times (np.ndarray): array of shape N
        power (np.ndarray): array of shape N
        impedance (np.ndarray): array of shape N
        temperature (np.ndarray): array of shape N
        force (AblationForce): data on the force at each ablation site. Specifically, the
            force applied, the time at which it was applied, and axial and lateral angles,
            and the 3D coordinates of the ablation site.
        auto_index (AblationAutoIndex): automatically acquired ablation site data from systems
            like Carto3 Visitag module
    """

    times: np.ndarray = None
    power: np.ndarray = None
    impedance: np.ndarray = None
    temperature: np.ndarray = None
    force: AblationForce = None
    auto_index: AblationAutoIndex = None

    def __attrs_post_init__(self):

        self.auto_index = AblationAutoIndex()
        if self.force is None:
            self.force = AblationForce()

    def __repr__(self):
        n_sites = {len(self.times)} if self.times is not None else 0
        return f"Ablations with {n_sites} ablation sites."

    def copy(self):
        """Create a deep copy of Ablation"""

        ablation = Ablation(
            times=np.array(self.times) if self.times is not None else None,
            power=np.array(self.power) if self.power is not None else None,
            impedance=np.array(self.impedance) if self.impedance is not None else None,
            temperature=np.array(self.temperature) if self.temperature is not None else None,
            force=self.force.copy(),
            auto_index=self.auto_index.copy(),
        )

        return ablation


def extract_ablation_data(ablation_data, rfindex_data=None):
    """Extract surface data from a dictionary.

    Args:
        ablation_data (dict): Dictionary containing numpy arrays that describe the
            ablation sites.

    Returns:
        ablation (Ablation): times, power, impedance and temperature for each ablation site,
            as well as the force applied.
    """
    ablation = Ablation()

    # Add rf_index data from openep.mat if present
    if rfindex_data is None or isinstance(rfindex_data, np.ndarray):
        ablation.auto_index = AblationAutoIndex()
    else:
        ablation.auto_index = AblationAutoIndex(
            times=rfindex_data['tag']['time'].astype(float),
            average_force=rfindex_data['tag']['avgForce'].astype(float),
            max_temp=rfindex_data['tag']['maxTemp'].astype(float),
            max_power=rfindex_data['tag']['maxPower'].astype(float),
            force_time_integral=rfindex_data['tag']['fti'].astype(float),
            points=rfindex_data['tag']['X'].astype(float)
        )

    if ablation_data is None or isinstance(ablation_data, np.ndarray):
        return ablation

    try:
        if not ablation_data['originaldata']['ablparams']['time'].size:
            return ablation
    except KeyError as e:
        return ablation
    except AttributeError as e:
        return ablation

    ablation.times = ablation_data['originaldata']['ablparams']['time'].astype(float)
    ablation.power = ablation_data['originaldata']['ablparams']['power'].astype(float)
    ablation.impedance = ablation_data['originaldata']['ablparams']['impedance'].astype(float)
    ablation.temperature = ablation_data['originaldata']['ablparams']['distaltemp'].astype(float)

    ablation.force = AblationForce(
        times=ablation_data['originaldata']['force']['time'].astype(float),
        force=ablation_data['originaldata']['force']['force'].astype(float),
        axial_angle=ablation_data['originaldata']['force']['axialangle'].astype(float),
        lateral_angle=ablation_data['originaldata']['force']['lateralangle'].astype(float),
        points=ablation_data['originaldata']['force']['position'].astype(float),
    )

    return ablation


def empty_ablation():
    """Create an empty Ablation object with empty numpy arrays.

    Returns:
        ablation (Ablation): times, power, impedance and temperature for each ablation site,
            as well as the force applied.
    """

    force = AblationForce(
        times=np.array([], dtype=float),
        force=np.array([], dtype=float),
        axial_angle=np.array([], dtype=float),
        lateral_angle=np.array([], dtype=float),
        points=np.array([], dtype=float),
    )

    ablation = Ablation(
        times=np.array([], dtype=float),
        power=np.array([], dtype=float),
        impedance=np.array([], dtype=float),
        temperature=np.array([], dtype=float),
        force=force,
    )

    return ablation
