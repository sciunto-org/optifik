import pytest
import warnings
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from optifik.minmax import thickness_from_minmax
from optifik.analysis import smooth_intensities
from optifik.io import load_spectrum

#
# Helper
#

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / 'data'


def test_minmax_ransac_data_basic(test_data_dir):
    spectrum_path = test_data_dir / 'basic' / '000004310.xy'
    expected = 1338.35

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  1.324188 + 3102.060378 / (lambdas**2)

    prominence = 0.02

    result = thickness_from_minmax(lambdas,
                                   smoothed_intensities,
                                   refractive_index=r_index,
                                   min_peak_prominence=prominence,
                                   method='ransac',
                                   plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)

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

def test_minmax_theory():
    lambda_min = 450
    lambda_max = 800
    lambdas = np.linspace(lambda_min, lambda_max, 1_000)
    h_values = np.linspace(300, 5_000, 33)

    n_values = n_lambda(lambdas)
    for expected in h_values:
        intensities = compute_spectrum_theory(expected, lambdas, n_values)

        result = thickness_from_minmax(lambdas,
                                       intensities,
                                       refractive_index=n_values,
                                       method='linreg',
                                       min_peak_prominence=None,
                                       plot=False)

        r_error = np.abs((result.thickness - expected) / expected)
        assert r_error < 1e-2


def test_minmax_less_than_2_peaks():
    lambda_min = 450
    lambda_max = 800
    lambdas = np.linspace(lambda_min, lambda_max, 1_000)
    h_values = np.linspace(100, 250, 3)

    n_values = n_lambda(lambdas)
    for expected in h_values:
        intensities = compute_spectrum_theory(expected, lambdas, n_values)
        with pytest.warns(RuntimeWarning):
            result = thickness_from_minmax(lambdas,
                                           intensities,
                                           refractive_index=n_values,
                                           method='linreg',
                                           min_peak_prominence=None,
                                           plot=False)

         
            assert result.thickness is np.nan




#
# Data
#

def test_minmax_linreg_data_basic(test_data_dir):
    spectrum_path = test_data_dir / 'basic' / '000004310.xy'
    expected = 1338.35

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  n_lambda(lambdas)

    prominence = 0.02

    result = thickness_from_minmax(lambdas,
                                   smoothed_intensities,
                                   refractive_index=r_index,
                                   min_peak_prominence=prominence,
                                   method='linreg',
                                   plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)


def test_minmax_linreg_data_basic(test_data_dir):
    spectrum_path = test_data_dir / 'basic' / '000004310.xy'
    expected = 1338.35

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  n_lambda(lambdas)

    prominence = 0.02

    result = thickness_from_minmax(lambdas,
                                   smoothed_intensities,
                                   refractive_index=r_index,
                                   min_peak_prominence=prominence,
                                   method='ransac',
                                   plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)
