import numpy as np
from scipy.optimize import curve_fit

import inspect
import matplotlib.pyplot as plt

from .io import load_spectrum
from .utils import OptimizeResult, setup_matplotlib
from .analysis import finds_peak


def _thicknesses_scheludko_at_order(wavelengths,
                                    intensities,
                                    interference_order,
                                    refractive_index,
                                    intensities_void=None):
    """
    Compute thicknesses vs wavelength for a given interference order.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    interference_order : int
        Interference order.
    refractive_index : array
        Refractive index.
    intensities_void : array, optional
        Intensities of void.

    Returns
    -------
    thicknesses : array

    """
    if intensities_void is None:
        Imin = np.min(intensities)
    else:
        Imin = intensities_void

    n = refractive_index
    m = interference_order
    I_norm = (np.asarray(intensities) - Imin) / (np.max(intensities) - Imin)

    prefactor = wavelengths / (2 * np.pi * n)
    argument = np.sqrt(I_norm / (1 + (1 - I_norm) * (n**2 - 1)**2 / (4 * n**2)))

    if m % 2 == 0:
        term1 = (m / 2) * np.pi
    else:
        term1 = ((m+1) / 2) * np.pi

    term2 = (-1)**m * np.arcsin(argument)

    return prefactor * (term1 + term2)


def _Delta(wavelengths, thickness, interference_order, refractive_index):
    """
    Compute the Delta values.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    thickness : array
        Film thickness.
    interference_order : int
        Interference order.
    refractive_index : array_like (or float)
        Refractive index.

    Returns
    -------
    ndarray
        Delta values.
    """

    # ensure that the entries are numpy arrays
    wavelengths = np.asarray(wavelengths)
    h = np.asarray(thickness)
    n = np.asarray(refractive_index)
    m = interference_order

    # Calculation of p as a function of the parity of m
    if m % 2 == 0:
        p = m / 2
    else:
        p = (m + 1) / 2

    # Calculation of alpha
    alpha = ((n**2 - 1)**2) / (4 * n**2)

    # Argument of sinus
    angle = (2 * np.pi * n * h / wavelengths) - p * np.pi

    # A = sin²(argument)
    A = np.sin(angle)**2

    # Final calcuation of Delta
    return (A * (1 + alpha)) / (1 + A * alpha)


def _Delta_fit(xdata, thickness, interference_order):
    """
    Wrapper on Delta() for curve_fit.

    Parameters
    ----------
    xdata : tuple
        (wavelengths, refractive_index)
    thickness : array_like (or float)
        Film thickness.
    interference_order : int
        Interference order.

    Returns
    -------
    ndarray
        Delta values.

    """
    lambdas, r_index = xdata
    return _Delta(lambdas, thickness, interference_order, r_index)


def get_default_start_stop_wavelengths(wavelengths,
                                       intensities,
                                       refractive_index,
                                       min_peak_prominence,
                                       plot=None):
    """
    Returns the start and stop wavelength values of the last monotonic branch.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    refractive_index : scalar or array
        Value of the refractive index of the medium.
    min_peak_prominence : scalar
        Required prominence of peaks.
    plot : bool, optional
        Display a curve, useful for checking or debuging. The default is None.


    Raises
    ------
    RuntimeError
        if at least one maximum and one minimum are not detected.

    Returns
    -------
    wavelength_start : scalar
    wavelength_stop : scalar
    """
    if isinstance(refractive_index, (float, int)):
        refractive_index  = np.full_like(wavelengths,  refractive_index)

    # idx_min idx max
    idx_peaks_min, idx_peaks_max = finds_peak(wavelengths, intensities,
                                              min_peak_prominence=min_peak_prominence,
                                              plot=plot)

    failure, message = False, ''
    if len(idx_peaks_min) == 0:
        message += 'Failed to detect at least one minimum. '
        failure = True
    if len(idx_peaks_max) == 0:
        message += 'Failed to detect at least one maximum. '
        failure = True
    if failure:
        raise RuntimeError(message)

    # Get the last oscillation peaks
    lambda_min = wavelengths[idx_peaks_min[-1]]
    lambda_max = wavelengths[idx_peaks_max[-1]]

    # Order them
    wavelength_start = min(lambda_min, lambda_max)
    wavelength_stop = max(lambda_min, lambda_max)

    return wavelength_start, wavelength_stop


def thickness_from_scheludko(wavelengths,
                             intensities,
                             refractive_index,
                             wavelength_start=None,
                             wavelength_stop=None,
                             interference_order=None,
                             intensities_void=None,
                             plot=None):
    """
    Compute the film thickness based on Scheludko method.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    refractive_index : scalar or array
        Value of the refractive index of the medium.
    wavelength_start : scalar, optional
        Starting value of a monotonic branch.
        Mandatory if interference_order != 0.
    wavelength_stop : scalar, optional
        Stoping value of a monotonic branch.
        Mandatory if interference_order != 0.
    interference_order : scalar, optional
        Interference order, zero or positive integer.
        If set to None, the value is guessed.
    intensities_void : array, optional
        Intensity in absence of a film.
        Mandatory if interference_order == 0.
    plot : bool, optional
        Display a curve, useful for checking or debuging. The default is None.

    Returns
    -------
    results : Instance of `OptimizeResult` class.
        The attribute `thickness` gives the thickness value in nm.

    """
    if isinstance(refractive_index, (float, int)):
        refractive_index  = np.full_like(wavelengths,  refractive_index)

    if plot:
        setup_matplotlib()

    if interference_order is None or interference_order > 0:
        if wavelength_stop is None or wavelength_start is None:
            raise ValueError('wavelength_start and wavelength_stop must be passed for interference_order != 0.')
        else:
            if wavelength_start > wavelength_stop:
                raise ValueError('wavelength_start and wavelength_stop are swapped.')

    r_index = refractive_index

    # Handle the interference order
    if interference_order is None:
        # A bit extreme...
        max_tested_order = 12

        # mask input data
        mask = (wavelengths >= wavelength_start) & (wavelengths <= wavelength_stop)
        wavelengths_masked = wavelengths[mask]
        r_index_masked = r_index[mask]
        intensities_masked = intensities[mask]

        min_difference = np.inf
        thickness_values = None

        if plot:
            plt.figure()
            plt.ylabel(r'$h$ $[\mathrm{{nm}}]$')
            plt.xlabel(r'$\lambda$ $[\mathrm{nm}]$')

        for _order in range(0, max_tested_order+1):
            h_values = _thicknesses_scheludko_at_order(wavelengths_masked,
                                                       intensities_masked,
                                                       _order,
                                                       r_index_masked)

            difference = np.max(h_values) - np.min(h_values)

            print(f"h-difference for m={_order}: {difference:.1f} nm")

            if difference < min_difference:
                min_difference = difference
                interference_order = _order
                thickness_values = h_values

            if plot:
                plt.plot(wavelengths_masked, h_values, '.',
                         markersize=3, label=f"Order={_order}")
    elif interference_order == 0:

        min_peak_prominence = 0.02
        peaks_min, peaks_max = finds_peak(wavelengths, intensities,
                                          min_peak_prominence=min_peak_prominence,
                                          plot=plot)
        if len(peaks_max) != 1:
            raise RuntimeError('Failed to detect a single maximum peak.')

        lambda_unique = wavelengths[peaks_max[0]]

        # keep rhs from the maximum
        mask = wavelengths >= lambda_unique
        wavelengths_masked = wavelengths[mask]
        r_index_masked = r_index[mask]
        intensities_masked = intensities[mask]
        intensities_void_masked = intensities_void[mask]

        interference_order = 0
        thickness_values = _thicknesses_scheludko_at_order(wavelengths_masked,
                                                     intensities_masked,
                                                     interference_order,
                                                     r_index_masked,
                                                     intensities_void=intensities_void_masked)

    elif interference_order > 0:
        # mask input data
        mask = (wavelengths >= wavelength_start) & (wavelengths <= wavelength_stop)
        wavelengths_masked = wavelengths[mask]
        r_index_masked = r_index[mask]
        intensities_masked = intensities[mask]

        h_values = _thicknesses_scheludko_at_order(wavelengths_masked,
                                                   intensities_masked,
                                                   interference_order,
                                                   r_index_masked)
        thickness_values = h_values
    else:
        raise ValueError('interference_order must be >= 0.')



    # Compute the thickness for the selected order

    # Delta
    if interference_order == 0:
        num = intensities_masked - np.min(intensities_void_masked)
        denom = np.max(intensities_masked) - np.min(intensities_void_masked)
    else:
        num = intensities_masked - np.min(intensities_masked)
        denom = np.max(intensities_masked) - np.min(intensities_masked)
    Delta_from_data = num / denom



    # Delta_from_data = (intensities_masked -np.min(intensities_masked))/(np.max(intensities_masked) -np.min(intensities_masked))
    # Delta_from_data = (intensities_raw_masked -np.min(intensities_raw_masked))/(np.max(intensities_raw_masked) -np.min(intensities_raw_masked))

    DeltaScheludko = _Delta(wavelengths_masked,
                            np.mean(thickness_values),
                            interference_order,
                            r_index_masked)


    xdata = (wavelengths_masked, r_index_masked)
    popt, pcov = curve_fit(lambda x, h: _Delta_fit(x, h, interference_order), xdata, Delta_from_data, p0=[np.mean(thickness_values)])
    fitted_h = popt[0]
    std_err = np.sqrt(pcov[0][0])

    if plot:
        Delta_values = _Delta(wavelengths_masked, fitted_h, interference_order, r_index_masked)

        plt.figure()
        plt.plot(wavelengths_masked, Delta_from_data,
                 'bo-', markersize=2,
                 label=r'$\mathrm{{Smoothed}}\ \mathrm{{Data}}$')

        # Scheludko
        label = rf'$\mathrm{{Scheludko}}\ (h = {np.mean(thickness_values):.1f} \pm {np.std(thickness_values):.1f}\ \mathrm{{nm}})$'
        plt.plot(wavelengths_masked, DeltaScheludko,
                 'go-', markersize=2, label=label)
        # Fit
        label = rf'$\mathrm{{Fit}}\ (h = {fitted_h:.1f}\pm {std_err:.1f} \ \mathrm{{nm}})$'
        plt.plot(wavelengths_masked,  Delta_values,
                 'ro-', markersize=2,
                 label=label)

        plt.legend()
        plt.ylabel(r'$\Delta$')
        plt.xlabel(r'$\lambda$ $[\mathrm{{nm}}]$')
        plt.title(f'Func Call: {inspect.currentframe().f_code.co_name}()')

    return OptimizeResult(thickness=fitted_h, stderr=std_err)

