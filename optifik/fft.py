import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft, fftfreq

import matplotlib.pyplot as plt

from .utils import OptimizeResult, setup_matplotlib


def thickness_from_fft(wavelengths, intensities,
                       refractive_index,
                       num_half_space=None,
                       plot=None):
    """
    Determine the tickness by Fast Fourier Transform.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    refractive_index : scalar or array
        Value of the refractive index of the medium.
    num_half_space : scalar, optional
        Number of points to compute FFT's half space.
        If `None`, default corresponds to `10*len(wavelengths)`.
    plot : boolean, optional
        Show plot of the transformed signal and the peak detection.

    Returns
    -------
    results : Instance of `OptimizeResult` class.
        The attribute `thickness` gives the thickness value in nm.
    """
    if num_half_space is None:
        num_half_space = 10 * len(wavelengths)

    x = refractive_index / wavelengths
    y = intensities

    # Resample the data
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
    x_uniform = np.linspace(x.min(), x.max(), 2 * num_half_space)
    density = x_uniform[1] - x_uniform[0]
    y_uniform = f(x_uniform)

    # FFT
    fft_values = fft(y_uniform)
    freqs = fftfreq(len(x_uniform), d=density)

    # Select positive side
    positive_freqs = freqs[freqs > 0]
    positive_fft = np.abs(fft_values[freqs > 0])

    # Find the prominent freq
    peak_index = np.argmax(positive_fft)
    optical_thickness = positive_freqs[peak_index]

    thickness = optical_thickness / 2.

    if plot:
        setup_matplotlib()
        plt.figure()
        plt.loglog(positive_freqs, positive_fft)
        plt.loglog(peak_index, optical_thickness, 'o')
        plt.xlabel('$\mathrm{{Optical \ Distance}} \ \mathcal{D}$ $[\mathrm{{nm}}]$')
        plt.ylabel(r'$\mathrm{{FFT}}$ $(I^\star)$')
        plt.title(f'Thickness={thickness:.2f}')

    return OptimizeResult(thickness=thickness,)
