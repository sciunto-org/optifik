import pytest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from optifik.scheludko import thickness_from_scheludko
from optifik.scheludko import get_default_start_stop_wavelengths
from optifik.analysis import smooth_intensities
from optifik.io import load_spectrum


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / 'data'


@pytest.fixture
def dataset1(test_data_dir):
    spectrum_path = test_data_dir / 'basic' / '000005253.xy'
    expected = 777.07

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index = 1.324188 + 3102.060378 / (lambdas**2)

    return {
        "expected": expected,
        "lambdas": lambdas,
        "smoothed_intensities": smoothed_intensities,
        "r_index": r_index,
    }



def test_data_interference_order_positive(dataset1):
    expected = dataset1['expected']
    lambdas = dataset1['lambdas']
    smoothed_intensities = dataset1['smoothed_intensities']
    r_index = dataset1['r_index']
    prominence = 0.02

    w_start, w_stop = 300, 500
    with pytest.raises(ValueError):
        result = thickness_from_scheludko(lambdas,
                                          smoothed_intensities,
                                          refractive_index=r_index,
                                          wavelength_start=w_start,
                                          wavelength_stop=w_stop,
                                          interference_order=-1,
                                          plot=False)

def test_data_start_stop_swapped(dataset1):
    expected = dataset1['expected']
    lambdas = dataset1['lambdas']
    smoothed_intensities = dataset1['smoothed_intensities']
    r_index = dataset1['r_index']
    prominence = 0.02

    w_start, w_stop = 500, 300
    with pytest.raises(ValueError):
        result = thickness_from_scheludko(lambdas,
                                          smoothed_intensities,
                                          refractive_index=r_index,
                                          wavelength_start=w_start,
                                          wavelength_stop=w_stop,
                                          interference_order=None,
                                          plot=False)


def test_data_scheludko_4peaks(dataset1):
    expected = dataset1['expected']
    lambdas = dataset1['lambdas']
    smoothed_intensities = dataset1['smoothed_intensities']
    r_index = dataset1['r_index']
    prominence = 0.02


    w_start, w_stop = get_default_start_stop_wavelengths(lambdas,
                                                         smoothed_intensities,
                                                         refractive_index=r_index,
                                                         min_peak_prominence=prominence,
                                                         plot=False)


    result = thickness_from_scheludko(lambdas,
                                      smoothed_intensities,
                                      refractive_index=r_index,
                                      wavelength_start=w_start,
                                      wavelength_stop=w_stop,
                                      interference_order=None,
                                      plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)


def test_data_scheludko_2peaks(test_data_dir):
    spectrum_path = test_data_dir / 'basic' / '000006544.xy'
    expected = 495.69

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  1.324188 + 3102.060378 / (lambdas**2)

    prominence = 0.03


    w_start, w_stop = get_default_start_stop_wavelengths(lambdas,
                                                         smoothed_intensities,
                                                         refractive_index=r_index,
                                                         min_peak_prominence=prominence,
                                                         plot=False)

    result = thickness_from_scheludko(lambdas,
                                      smoothed_intensities,
                                      refractive_index=r_index,
                                      wavelength_start=w_start,
                                      wavelength_stop=w_stop,
                                      interference_order=None,
                                      plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)


def test_data_order0(test_data_dir):
    spectrum_path = test_data_dir / 'basic' / '000018918.xy'
    expected = 115.33

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.03


    File_I_min = test_data_dir / 'basic' / 'void.xy'
    _, intensities_void = load_spectrum(File_I_min, wavelength_min=450)


    w_start, w_stop = None, None
    result = thickness_from_scheludko(lambdas,
                                      smoothed_intensities,
                                      refractive_index=r_index,
                                      wavelength_start=w_start,
                                      wavelength_stop=w_stop,
                                      interference_order=0,
                                      intensities_void=intensities_void,
                                      plot=False)


    assert_allclose(result.thickness, expected, rtol=1e-1)


