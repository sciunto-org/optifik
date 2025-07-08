import pytest
import yaml
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from optifik.scheludko import thickness_from_scheludko
from optifik.scheludko import get_default_start_stop_wavelengths
from optifik.io import load_spectrum
from optifik.analysis import smooth_intensities


def load(filename):
    test_data_dir = Path(__file__).parent.parent / 'data'
    FOLDER = test_data_dir / 'spectraVictor2' / 'order0'

    yaml_file = FOLDER / filename
    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)
    data = [(FOLDER / fn, val) for fn, val in thickness_dict.items()]
    return data


#@pytest.mark.skip('...')
@pytest.mark.parametrize("spectrum_path, expected", load('known_value.yaml'))
def test_SV2o0_small_tol(spectrum_path, expected):
    test_data_dir = Path(__file__).parent.parent / 'data'

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)

    File_I_min = test_data_dir / 'spectraVictor2' / 'void.xy'
    _, intensities_void = load_spectrum(File_I_min, wavelength_min=450)

    smoothed_intensities = smooth_intensities(raw_intensities)

    r_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.020

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

@pytest.mark.parametrize("spectrum_path, expected", load('known_value_large_tol.yaml'))
def test_SV2o0_large_tol(spectrum_path, expected):
    test_data_dir = Path(__file__).parent.parent / 'data'

    lambdas, raw_intensities = load_spectrum(spectrum_path, wavelength_min=450)

    File_I_min = test_data_dir / 'spectraVictor2' / 'void.xy'
    _, intensities_void = load_spectrum(File_I_min, wavelength_min=450)

    smoothed_intensities = smooth_intensities(raw_intensities)

    r_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.020

    w_start, w_stop = None, None
    result = thickness_from_scheludko(lambdas,
                                      smoothed_intensities,
                                      refractive_index=r_index,
                                      wavelength_start=w_start,
                                      wavelength_stop=w_stop,
                                      interference_order=0,
                                      intensities_void=intensities_void,
                                      plot=False)

    assert_allclose(result.thickness, expected, rtol=2.5e-1)

