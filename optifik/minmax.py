import numpy as np

from scipy import stats
from skimage.measure import ransac, LineModelND
from scipy.signal import find_peaks

import inspect
import matplotlib.pyplot as plt

from .utils import OptimizeResult, setup_matplotlib


def thickness_from_minmax(wavelengths,
                          intensities,
                          refractive_index,
                          min_peak_prominence,
                          min_peak_distance=10,
                          method='linreg',
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
        # Can't fit if less than two points.
        return OptimizeResult(thickness=np.nan)


    if isinstance(refractive_index, np.ndarray):
        #refractive_index = refractive_index[peaks][::-1]
        n_over_lambda = refractive_index[peaks][::-1] / wavelengths[peaks][::-1]
    else:
        n_over_lambda = refractive_index / wavelengths[peaks][::-1]

    if method.lower() == 'ransac':
        residual_threshold = 4e-5
        min_samples = 2
        # Scikit-image
        data = np.column_stack([k_values, n_over_lambda])
        model_robust, inliers = ransac(data, LineModelND,
                                       min_samples=min_samples,
                                       residual_threshold=residual_threshold,
                                       max_trials=100)
        slope = model_robust.params[1][1]
        thickness_minmax = 1 / slope /  4

        # Scikit-learn
        #from sklearn import linear_model, datasets
        #X, y = k_values.reshape(-1, 1), n_over_lambda[::-1]

        ## Fit line using all data
        #lr = linear_model.LinearRegression()
        #lr.fit(X, y)

        #slransac = linear_model.RANSACRegressor(min_samples=min_samples,
        #                                        residual_threshold=residual_threshold)
        #slransac.fit(X, y)
        #inlier_mask = slransac.inlier_mask_
        #outlier_mask = np.logical_not(inlier_mask)

        ## Predict data of estimated models
        #line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        #line_y = lr.predict(line_X)
        #line_y_ransac = slransac.predict(line_X)

        #slope = -slransac.estimator_.coef_[0]
        #thickness_minmax = 1 / slope /  4

        if plot:
            fig, ax = plt.subplots()

            ax.set_xlabel(r'$\mathrm{{Index}}$ $N$')
            ax.set_ylabel('$n$($\lambda$) / $\lambda$ \ $[\mathrm{{\mu m^{-1}}}]$ ')
            ax.plot(data[inliers, 0], data[inliers, 1] * 1000, 'xb', alpha=0.6, label='Inliers')
            ax.plot(data[~inliers, 0], data[~inliers, 1] * 1000, '+r', alpha=0.6, label='Outliers')
            ax.plot(k_values, model_robust.predict_y(k_values) * 1000, '-g', label='Fit')

            ax.legend()
            ax.set_title(f'Thickness = {thickness_minmax:.2f} nm')
            plt.title(f'Func Call: {inspect.currentframe().f_code.co_name}()')
            plt.tight_layout()
            plt.show()

        return OptimizeResult(thickness=thickness_minmax,
                              num_inliers=inliers.sum(),
                              num_outliers=(~inliers).sum(),
                              peaks_max=peaks_max,
                              peaks_min=peaks_min)

    elif method.lower() == 'linreg':
        slope, intercept, r_value, p_value, std_err = stats.linregress(k_values, n_over_lambda)
        #mean_n = np.mean(refractive_index)
        thickness_minmax = 1 / slope / 4

        if plot:
            fig, ax = plt.subplots()

            ax.set_xlabel(r'$\mathrm{{Index}}$ $N$')
            ax.set_ylabel('$n$($\lambda$) / $\lambda$ \ $[\mathrm{{\mu m^{-1}}}]$ ')
            ax.plot(k_values, n_over_lambda * 1000, 's', label='Extrema')
            ax.plot(k_values, (intercept + k_values * slope) * 1000, label='Fit')

            ax.legend()
            ax.set_title(f'Thickness = {thickness_minmax:.2f} nm')
            plt.title(f'Func Call: {inspect.currentframe().f_code.co_name}()')
            plt.tight_layout()
            plt.show()

        return OptimizeResult(thickness=thickness_minmax,
                              peaks_max=peaks_max,
                              peaks_min=peaks_min,
                              stderr=std_err)

    else:
        raise ValueError('Wrong method')
