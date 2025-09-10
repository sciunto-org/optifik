import pytest
import yaml
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from optifik.fft import thickness_from_fft
from optifik.analysis import smooth_intensities
from optifik.io import load_spectrum


def load_Lorene():
    test_data_dir = Path(__file__).parent.parent / 'data'
    FOLDER = test_data_dir / 'spectraLorene/sample2'
    yaml_file = FOLDER / 'sample2.yaml'

    with open(yaml_file, "r") as yaml_file:
        content = yaml.safe_load(yaml_file)
        thickness_dict = content['known_thicknesses']

    data = [(FOLDER / fn, val) for fn, val in thickness_dict.items()]
    return data

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / 'data'


def test_FFT_data_basic(test_data_dir):
    spectrum_path = test_data_dir / 'basic' / '003582.xy'
    expected = 3524.51

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  1.324188 + 3102.060378 / (lambdas**2)

    thickness_FFT = thickness_from_fft(lambdas,
                                       smoothed_intensities,
                                       refractive_index=r_index)
    result = thickness_FFT.thickness

    assert_allclose(result, expected, rtol=1e-1)

@pytest.mark.parametrize("spectrum_path, expected", load_Lorene())
def test_FFT_data_Lorene(spectrum_path, expected):
    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=680)
    smoothed_intensities = smooth_intensities(raw_intensities)
    smoothed_intensities = raw_intensities.copy()

    r_index = 1.41
    thickness_FFT = thickness_from_fft(lambdas,
                                       smoothed_intensities,
                                       refractive_index=r_index,
                                       plot=True
                                       )


    result = thickness_FFT.thickness
    assert_allclose(result, expected, rtol=1e-1)
