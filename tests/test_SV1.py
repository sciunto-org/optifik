import pytest
import yaml
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from optifik.minmax import thickness_from_minmax
from optifik.io import load_spectrum
from optifik.analysis import smooth_intensities
from optifik.analysis import finds_peak


def load():
    test_data_dir = Path(__file__).parent.parent / 'data'
    FOLDER = test_data_dir / 'spectraVictor1'
    yaml_file = FOLDER / 'known_value.yaml'

    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)

    data = [(FOLDER / fn, val) for fn, val in thickness_dict.items()]
    return data


@pytest.mark.parametrize("spectrum_path, expected", load())
def test_minmax(spectrum_path, expected):
    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)

    assert_equal(len(lambdas), len(smoothed_intensities))

    indice =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.02

    thickness_minmax = thickness_from_minmax(lambdas,
                                             smoothed_intensities,
                                             refractive_index=indice,
                                             min_peak_prominence=prominence)

    result = thickness_minmax.thickness

    assert_allclose(result, expected, rtol=1e-1)

