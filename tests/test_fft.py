import pytest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from optifik.fft import thickness_from_fft
from optifik.analysis import smooth_intensities
from optifik.io import load_spectrum

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / 'data'


def test_FFT(test_data_dir):
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
