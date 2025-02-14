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

import pytest
from numpy.testing import assert_allclose

import numpy as np

import openep
from openep._datasets.openep_datasets import DATASET_2


@pytest.fixture()
def case():
    return openep.load_openep_mat(DATASET_2)


@pytest.fixture()
def exported_case(case, tmp_path):

    EXPORTED_CASE = tmp_path / "exported_case.mat"
    openep.export_openep_mat(case, EXPORTED_CASE.as_posix())

    return openep.load_openep_mat(EXPORTED_CASE)

@pytest.fixture
def output_filename(tmp_path):
    return tmp_path / "_export_case.mat"


def test_openep_mat_export(case, exported_case):
    """Check the scalar fields of the original and exported data set are equal."""

    assert np.all(case.notes == exported_case.notes)

    assert_allclose(case.points, exported_case.points)
    assert_allclose(case.indices, exported_case.indices)

    assert_allclose(case.fields.bipolar_voltage, exported_case.fields.bipolar_voltage, equal_nan=True)
    assert_allclose(case.fields.unipolar_voltage, exported_case.fields.unipolar_voltage, equal_nan=True)
    assert_allclose(case.fields.local_activation_time, exported_case.fields.local_activation_time, equal_nan=True)
    assert exported_case.fields.force is None
    assert exported_case.fields.impedance is None
    assert exported_case.fields.thickness is None
    assert exported_case.fields.cell_region is None
    assert exported_case.fields.longitudinal_fibres is None
    assert exported_case.fields.transverse_fibres is None

    assert np.all(case.electric.names == exported_case.electric.names)
    assert np.all(case.electric.internal_names == exported_case.electric.internal_names)
    assert_allclose(case.electric.include, exported_case.electric.include)

    assert_allclose(case.electric.bipolar_egm.egm, exported_case.electric.bipolar_egm.egm)
    assert_allclose(case.electric.bipolar_egm.points, exported_case.electric.bipolar_egm.points)
    assert_allclose(case.electric.bipolar_egm.voltage, exported_case.electric.bipolar_egm.voltage)
    assert_allclose(case.electric.bipolar_egm.gain, exported_case.electric.bipolar_egm.gain)
    assert np.all(case.electric.bipolar_egm.names == exported_case.electric.bipolar_egm.names)

    assert_allclose(case.electric.unipolar_egm.egm, exported_case.electric.unipolar_egm.egm)
    assert_allclose(case.electric.unipolar_egm.points, exported_case.electric.unipolar_egm.points)
    assert_allclose(case.electric.unipolar_egm.voltage, exported_case.electric.unipolar_egm.voltage)
    assert_allclose(case.electric.unipolar_egm.gain, exported_case.electric.unipolar_egm.gain)
    assert np.all(case.electric.unipolar_egm.names == exported_case.electric.unipolar_egm.names)

    assert_allclose(case.electric.reference_egm.egm, exported_case.electric.reference_egm.egm)
    assert_allclose(case.electric.reference_egm.gain, exported_case.electric.reference_egm.gain)
    assert exported_case.electric.reference_egm.points is None
    assert_allclose(case.electric.reference_egm.voltage, exported_case.electric.reference_egm.voltage, equal_nan=True)
    assert all(exported_case.electric.reference_egm.names == ' ')

    assert_allclose(case.electric.ecg.ecg, exported_case.electric.ecg.ecg)
    assert_allclose(case.electric.ecg.gain, exported_case.electric.ecg.gain)
    assert np.all(case.electric.ecg.channel_names == exported_case.electric.ecg.channel_names)

    assert_allclose(case.electric.annotations.reference_activation_time, case.electric.annotations.reference_activation_time)
    assert_allclose(case.electric.annotations.local_activation_time, case.electric.annotations.local_activation_time)
    assert_allclose(case.electric.annotations.window_of_interest, case.electric.annotations.window_of_interest)

    assert_allclose(case.ablation.times, exported_case.ablation.times)
    assert_allclose(case.ablation.power, exported_case.ablation.power)
    assert_allclose(case.ablation.impedance, exported_case.ablation.impedance)
    assert_allclose(case.ablation.temperature, exported_case.ablation.temperature)

    assert_allclose(case.ablation.force.times, exported_case.ablation.force.times)
    assert_allclose(case.ablation.force.points, exported_case.ablation.force.points)
    assert_allclose(case.ablation.force.force, exported_case.ablation.force.force)
    assert_allclose(case.ablation.force.axial_angle, exported_case.ablation.force.axial_angle)
    assert_allclose(case.ablation.force.lateral_angle, exported_case.ablation.force.lateral_angle)


def test_additional_fields_export(case, output_filename):
    """Check if additional fields are exported then imported correctly"""

    mesh = case.create_mesh()

    case.fields['hello'] = case.fields.bipolar_voltage
    case.fields['world'] = case.fields.unipolar_voltage
    case.fields['empty'] = np.full(mesh.n_points, fill_value=np.NaN)
    case.fields['none'] = None

    openep.export_openep_mat(case,  output_filename)
    ts_case = openep.load_openep_mat(output_filename)

    np.testing.assert_array_equal(ts_case.fields.hello, case.fields.hello)
    np.testing.assert_array_equal(ts_case.fields.world, case.fields.world)
    np.testing.assert_array_equal(ts_case.fields.empty, case.fields.empty)

    assert 'none' not in ts_case.fields
    assert 'none' not in ts_case.fields.CUSTOM_FIELD_NAME

def test_opencarp_mat_export(case, tmp_path):
    path_prefix = tmp_path / "exported_case"

    openep.export_openCARP(
        case=case,
        prefix=path_prefix,
        scale_points=1000,  # convert mm to micrometre
        export_transverse_fibres=False,  # TODO: this should be a user-preference
    )

    carp_case = openep.load_opencarp(
        points=f"{path_prefix}.pts",
        indices=f"{path_prefix}.elem",
        fibres=f"{path_prefix}.lon",
        scale_points=1e-3,
    )

    openep_mesh = case.create_mesh()
    opencarp_mesh = carp_case.create_mesh()

    assert openep_mesh.n_points == opencarp_mesh.n_points
    assert openep_mesh.n_faces == opencarp_mesh.n_faces
    assert np.allclose(openep_mesh.points, opencarp_mesh.points)
    assert np.array_equal(openep_mesh.faces, opencarp_mesh.faces)

    assert np.array_equal(case.vectors.fibres, carp_case.vectors.fibres)
    assert np.array_equal(case.vectors.linear_connections, carp_case.vectors.linear_connections)
    assert np.array_equal(case.vectors.linear_connection_regions, carp_case.vectors.linear_connection_regions)
