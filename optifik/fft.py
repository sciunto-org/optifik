import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq

import inspect
import matplotlib.pyplot as plt

from .utils import OptimizeResult, setup_matplotlib, round_to_uncertainty

def thickness_from_fft(wavelengths, intensities,
                       refractive_index,
                       N_padding=1,
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
    N_padding : int, optional
        Multiply the space by `N_padding` with zero-padding.
        This can be used to refine the peak detection.
        Default: 1.
    num_half_space : scalar, optional
        Number of points to compute FFT's half space.
        If `None`, default corresponds to `10*len(wavelengths)`.
    plot : boolean, optional
        Show plot of the transformed signal and the peak detection.

    Returns
    -------
    results : Instance of `OptimizeResult` class.
        The attribute `thickness` gives the thickness value in nm.

    Notes
    -----
    if `N_padding` > 1, the peak is first detected without zero-padding,
    ie `N_padding` = 1. Then, padding is applied and the detection
    is done nearby the first peak detection.
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
    # First step, no padding
    fft_values = fft(y_uniform)
    freqs = fftfreq(len(x_uniform), d=density)

    # Select positive side
    positive_freqs = freqs[freqs > 0]
    positive_fft = np.abs(fft_values[freqs > 0])

    # Find the prominent freq
    peak_index = np.argmax(positive_fft)
    optical_thickness = positive_freqs[peak_index]

    thickness = optical_thickness / 2.
    error = np.diff(positive_freqs)[0]



    if N_padding > 1:
        fft_values = fft(y_uniform, n=N_padding*len(x_uniform))
        freqs = fftfreq(N_padding*len(x_uniform), d=density)

        # Subset around the main peak
        mask_peak = (freqs < optical_thickness + 2 * error) & (freqs > optical_thickness - 2 * error)
        positive_freqs_padding = freqs[mask_peak]
        positive_fft_padding = np.abs(fft_values[mask_peak])

        # Find the prominent freq
        peak_index_padding = np.argmax(positive_fft_padding)
        optical_thickness = positive_freqs_padding[peak_index_padding]

        thickness = optical_thickness / 2.
        error = np.diff(positive_freqs_padding)[0]

    if plot:
        setup_matplotlib()
        if N_padding > 1:
            fig, ax = plt.subplots(nrows=2)
            ax[0].set_title(f'Func Call: {inspect.currentframe().f_code.co_name}()')
            ax[0].loglog(positive_freqs, positive_fft)
            ax[0].loglog(optical_thickness, positive_fft[peak_index], 'o')

            val, err = round_to_uncertainty(thickness, error)
            label = rf'$\mathrm{{Scheludko}}\ (h = {val} \pm {err}\ \mathrm{{nm}})$'
            ax[1].set_title(f'Padding')
            ax[1].loglog(positive_freqs_padding, positive_fft_padding)
            ax[1].loglog(optical_thickness, positive_fft_padding[peak_index_padding],
                         'o', label=label)
            plt.legend()

            for a in ax:
                a.set_xlabel(r'$\mathrm{{Optical \ Distance}} \ \mathcal{D}$ $[\mathrm{{nm}}]$')
                a.set_ylabel(r'$\mathrm{{FFT}}$ $(I^\star)$')
        else:
            plt.figure()
            plt.loglog(positive_freqs, positive_fft)

            val, err = round_to_uncertainty(thickness, error)
            label = rf'$\mathrm{{Scheludko}}\ (h = {val} \pm {err}\ \mathrm{{nm}})$'
            plt.loglog(optical_thickness, positive_fft[peak_index], 'o', label=label)
            plt.xlabel(r'$\mathrm{{Optical \ Distance}} \ \mathcal{D}$ $[\mathrm{{nm}}]$')
            plt.ylabel(r'$\mathrm{{FFT}}$ $(I^\star)$')
            plt.title(f'Func Call: {inspect.currentframe().f_code.co_name}()')
            plt.legend()

    return OptimizeResult(thickness=thickness,
                          thickness_uncertainty=error)
