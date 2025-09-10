import pytest
import yaml
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from optifik.fft import thickness_from_fft
from optifik.analysis import smooth_intensities
from optifik.io import load_spectrum

#
# Helper
#

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


def n_lambda(lmbda):
    """
    For water + TTAB 1 CMC
    """
    return 1.324188 + 3102.060378 / (lmbda**2)

def compute_spectrum_theory(h, lambdas, n_values):
    sin_term = np.sin(2 * np.pi * n_values * h / lambdas) ** 2
    denominator = (2 * n_values / (n_values**2 - 1)) ** 2 + sin_term
    return sin_term / denominator


#
# Theory
#

def test_FFT_theory_range1():
    lambda_min = 450
    lambda_max = 800
    lambdas = np.linspace(lambda_min, lambda_max, 10_000)
    h_values = np.linspace(1_500, 5_000, 33)

    n_values = n_lambda(lambdas)
    for expected in h_values:
        intensities = compute_spectrum_theory(expected, lambdas, n_values)

        result = thickness_from_fft(lambdas,
                                    intensities,
                                    refractive_index=n_values,
                                    num_half_space=None,
                                    plot=False)

        r_error = np.abs((result.thickness - expected) / expected)
        assert r_error < 0.12


def test_FFT_theory_range2():
    lambda_min = 450
    lambda_max = 800
    lambdas = np.linspace(lambda_min, lambda_max, 10_000)
    h_values = np.linspace(5_000, 20_000, 33)

    n_values = n_lambda(lambdas)
    for expected in h_values:
        intensities = compute_spectrum_theory(expected, lambdas, n_values)

        result = thickness_from_fft(lambdas,
                                    intensities,
                                    refractive_index=n_values,
                                    num_half_space=None,
                                    plot=False)

        r_error = np.abs((result.thickness - expected) / expected)
        assert r_error < 4e-2


#
# Data
#

def test_FFT_data_basic(test_data_dir):
    spectrum_path = test_data_dir / 'basic' / '003582.xy'
    expected = 3524.51

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  n_lambda(lambdas)

    result = thickness_from_fft(lambdas,
                                       smoothed_intensities,
                                       refractive_index=r_index,
                                       plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)


@pytest.mark.parametrize("spectrum_path, expected", load_Lorene())
def test_FFT_data_Lorene(spectrum_path, expected):
    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=680)
    smoothed_intensities = smooth_intensities(raw_intensities)
    smoothed_intensities = raw_intensities.copy()

    r_index = 1.41
    result = thickness_from_fft(lambdas,
                                smoothed_intensities,
                                refractive_index=r_index,
                                plot=False)


    assert_allclose(result.thickness, expected, rtol=1e-1)
