import warnings
import numpy as np

from scipy import stats
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.signal import find_peaks

import inspect
import matplotlib.pyplot as plt

from .utils import OptimizeResult, setup_matplotlib, round_to_uncertainty


def thickness_from_minmax(wavelengths,
                          intensities,
                          refractive_index,
                          min_peak_prominence,
                          min_peak_distance=10,
                          method='linreg',
                          ransac_residual_threshold=1e-4,
                          plot=None):

    """
    Return the thickness from a min-max detection.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    refractive_index : scalar, optional
        Value of the refractive index of the medium.
    min_peak_prominence : scalar, optional
        Required prominence of peaks.
    min_peak_distance : scalar, optional
        Minimum distance between peaks.
    method : string, optional
        Either 'linreg' for linear regression or 'ransac'
        for Randon Sampling Consensus.
    ransac_residual_threshold : float, optional
        Residual threshold for ransac.
        Used only if `method=='ransac'`.
    plot : boolean, optional
        Show plots of peak detection and lin regression.

    Returns
    -------
    results : Instance of `OptimizeResult` class.
        The attribute `thickness` gives the thickness value in nm.

    Notes
    -----
    For more details about `min_peak_prominence` and `min_peak_distance`,
    see the documentation of `scipy.signal.find_peaks`. This function
    is used to find extrema.
    """
    if plot:
        setup_matplotlib()

    peaks_max, _ = find_peaks(intensities, prominence=min_peak_prominence, distance=min_peak_distance)
    peaks_min, _ = find_peaks(-intensities, prominence=min_peak_prominence, distance=min_peak_distance)
    peaks = np.concatenate((peaks_min, peaks_max))
    peaks.sort()

    k_values = np.arange(len(peaks))

    if k_values.size < 2:
        warnings.warn('Number of peaks < 2, cannot fit. Thickness set to NaN.', RuntimeWarning)
        return OptimizeResult(thickness=np.nan)

    if isinstance(refractive_index, np.ndarray):
        n_over_lambda = refractive_index[peaks][::-1] / wavelengths[peaks][::-1]
    else:
        n_over_lambda = refractive_index / wavelengths[peaks][::-1]

    if method.lower() == 'ransac':
        min_samples = 2
        data = np.column_stack([k_values, n_over_lambda])

        # Scikit-image
        #from skimage.measure import ransac, LineModelND
        #
        #model_robust, inliers = ransac(data, LineModelND,
        #                               min_samples=min_samples,
        #                               residual_threshold=residual_threshold,
        #                               max_trials=100)
        #slope = model_robust.params[1][1]
        #thickness_minmax = 1 / slope /  4

        # Organize the data for RANSAC (sklearn)
        X = k_values.reshape(-1, 1)
        y = n_over_lambda

        # Fit
        model_robust = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=min_samples,
            residual_threshold=ransac_residual_threshold,
            max_trials=100
        )
        model_robust.fit(X, y)

        inliers = model_robust.inlier_mask_
        slope = model_robust.estimator_.coef_[0]
        thickness_minmax = 1 / slope / 4

        # STD
        x_in = data[inliers, 0]
        y_in = data[inliers, 1]
        slope, intercept, r_value, p_value, slope_err = stats.linregress(x_in, y_in)
        thickness_err = slope_err / (4 * slope**2)


        if plot:
            val, err = round_to_uncertainty(thickness_minmax, thickness_err)
            label = rf'$\mathrm{{Fit}}\ (h = {val} \pm {err}\ \mathrm{{nm}})$'

            fig, ax = plt.subplots()
            ax.set_xlabel(r'$\mathrm{{Index}}$ $N$')
            ax.set_ylabel(r'$n$($\lambda$) / $\lambda$ \ $[\mathrm{{\mu m^{-1}}}]$ ')
            ax.plot(data[inliers, 0], data[inliers, 1] * 1000, 'xb', alpha=0.6, label='Inliers')
            ax.plot(data[~inliers, 0], data[~inliers, 1] * 1000, '+r', alpha=0.6, label='Outliers')
            ax.plot(k_values, model_robust.predict(X) * 1000, '-g', label=label)

            ax.legend()
            plt.title(f'Func Call: {inspect.currentframe().f_code.co_name}()')
            plt.tight_layout()
            plt.show()

        return OptimizeResult(thickness=thickness_minmax,
                              num_inliers=inliers.sum(),
                              num_outliers=(~inliers).sum(),
                              peaks_max=peaks_max,
                              peaks_min=peaks_min,
                              thickness_uncertainty=thickness_err)

    elif method.lower() == 'linreg':
        slope, intercept, r_value, p_value, slope_err = stats.linregress(k_values, n_over_lambda)
        thickness_minmax = 1 / slope / 4
        thickness_err = slope_err / (4 * slope**2)

        if plot:
            val, err = round_to_uncertainty(thickness_minmax, thickness_err)
            label = rf'$\mathrm{{Fit}}\ (h = {val} \pm {err}\ \mathrm{{nm}})$'

            fig, ax = plt.subplots()
            ax.set_xlabel(r'$\mathrm{{Index}}$ $N$')
            ax.set_ylabel(r'$n$($\lambda$) / $\lambda$ \ $[\mathrm{{\mu m^{-1}}}]$ ')
            ax.plot(k_values, n_over_lambda * 1000, 's', label='Extrema')
            ax.plot(k_values, (intercept + k_values * slope) * 1000, label=label)

            ax.legend()

            plt.title(f'Func Call: {inspect.currentframe().f_code.co_name}()')
            plt.tight_layout()
            plt.show()

        return OptimizeResult(thickness=thickness_minmax,
                              peaks_max=peaks_max,
                              peaks_min=peaks_min,
                              thickness_uncertainty=thickness_err)

    else:
        raise ValueError('Wrong method')
