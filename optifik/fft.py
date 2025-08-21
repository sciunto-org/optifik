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
    refractive_index : scalar, optional
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
    if plot:
        setup_matplotlib()

    if num_half_space is None:
        num_half_space = 10 * len(wavelengths)

    # FFT requires evenly spaced data.
    # So, we interpolate the signal.
    # Interpolate to get a linear increase of r_index / wavelengths.
    inverse_wavelengths_times_n = refractive_index / wavelengths
    f = interp1d(inverse_wavelengths_times_n, intensities)

    inverse_wavelengths_linspace = np.linspace(inverse_wavelengths_times_n[0],
                                           inverse_wavelengths_times_n[-1],
                                           2*num_half_space)
    intensities_linspace = f(inverse_wavelengths_linspace)


    # Perform FFT
    density = (inverse_wavelengths_times_n[-1]-inverse_wavelengths_times_n[0]) / (2*num_half_space)
    inverse_wavelengths_fft = fftfreq(2*num_half_space, density)
    intensities_fft = fft(intensities_linspace)

    # The FFT is symetrical over [0:N] and [N:2N].
    # Keep over [N:2N], ie for positive freq.
    intensities_fft = intensities_fft[num_half_space:2*num_half_space]
    inverse_wavelengths_fft = inverse_wavelengths_fft[num_half_space:2*num_half_space]

    idx_max_fft = np.argmax(abs(intensities_fft))
    freq_max = inverse_wavelengths_fft[idx_max_fft]

    thickness_fft = freq_max / 2.

    if plot:
        plt.figure()
        plt.loglog(inverse_wavelengths_fft, np.abs(intensities_fft))
        plt.loglog(freq_max, np.abs(intensities_fft[idx_max_fft]), 'o')
        plt.xlabel('$\mathrm{{Optical \ Distance}} \ \mathcal{D}$ $[\mathrm{{nm}}]$')
        plt.ylabel(r'$\mathrm{{FFT}}$ $(I^\star)$')
        plt.title(f'Thickness={thickness_fft:.2f}')

    return OptimizeResult(thickness=thickness_fft,)
