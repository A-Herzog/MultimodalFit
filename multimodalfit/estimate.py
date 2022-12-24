"""Multimodal distribution fitter."""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gamma
from scipy.optimize import fsolve, minimize
from .tools import indicators


__title__ = "multimodalfit"
__version__ = "1.0"
__author__ = "Alexander Herzog"
__email__ = "alexander.herzog@tu-clausthal.de"
__copyright__ = """
Copyright 2022 Alexander Herzog

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
__license__ = "Apache 2.0"


def dist_lognormal(mean: float, sd: float) -> object:
    """Generates a frozen distribution object for the log-normal distribution.

    Args:
        mean (float): Mean of the distribution
        sd (float): Standard deviation of the distribution

    Returns:
        object: Frozen distribution object
    """
    mu = math.log(mean**2 / math.sqrt(sd**2 + mean**2))
    sigma = math.sqrt(math.log((sd**2 / mean**2) + 1))
    return lognorm(sigma, scale=math.exp(mu))


def dist_gamma(mean: float, sd: float) -> object:
    """Generates a frozen distribution object for the gamma distribution.

    Args:
        mean (float): Mean of the distribution
        sd (float): Standard deviation of the distribution

    Returns:
        object: Frozen distribution object
    """
    beta = mean / (sd * sd)
    alpha = mean * beta
    beta = 1 / beta
    return gamma(alpha, loc=0, scale=beta)


def estimate_lognormal_mean(mode: float, mean: float, sd: float) -> float:
    """Estimates the mean of a log-normal distribution based on mode and standard deviation.

    Args:
        mode (float): Mode of the log-normal distribution
        mean (float): Start value for iterative calculation of the mean
        sd (float): Standard deviation of the log-normal distribution

    Returns:
        float: Estimated mean of the distribution
    """
    def f(x): return x**4 / math.sqrt(sd**2 + x**2) / (sd**2 + x**2) - mode

    roots = fsolve(f, x0=mean)
    return roots[0]


def estimate_gamma_mean(mode: float, mean: float, sd: float) -> float:
    """Estimates the mean of a gamma distribution based on mode and standard deviation.

    Args:
        mode (float): Mode of the gamma distribution
        mean (float): Start value for iterative calculation of the mean
        sd (float): Standard deviation of the gamma distribution

    Returns:
        float: Estimated mean of the distribution
    """
    if mode <= 0.0001:
        return mean

    def f(x): return (x**2 - sd**2) / x - mode

    roots = fsolve(f, x0=mean)
    return roots[0]


def estimation_step(hist: np.ndarray, estimate_mean: callable, fit_distribution: callable, show_info: bool = False, show_plot: bool = False, force_sd: float = -1.0, is_first_step: bool = False, fraction_all: float = 1) -> tuple[object, np.ndarray]:
    """Performs an estimation step (removes on distribution from the histogram values).

    Args:
        hist (np.ndarray): Histogram values where the distribution is to be found in.
        estimate_mean (callable): Function for estimating the mean value from mode and standard deviation
        fit_distribution (callable): Function for getting a frozen distribution object from given mean and standard deviation
        show_info (bool, optional): Print out information on the partial distribution? Defaults to False.
        show_plot (bool, optional): Generate diagram of histogram, found partial distribution and residuum? Defaults to False.
        force_sd (float, optional): If a value >=0 is passed here, this value will be used as standard deviation (instead of the real standard deviation of the histogram) in the mean estimation process. Defaults to -1.0.
        is_first_step (bool, optional): Is this the first partial distribution finding step? Defaults to False.
        fraction_all (float, optional): Used only in information printer. The fraction of the histogram given as parameter to the whole histogram. (For example is always 1 for the first step.) Defaults to 1.

    Returns:
        tuple[object, np.ndarray]: Parameters of the found distribution ("mean", "mode", "sd" and "fraction") and the residuum
    """

    # Fix histogram values if needed (make all values >=0)
    histFixed = np.where(hist >= 0, hist, 0)
    histFixed_sum = np.sum(histFixed)

    hist_mode, hist_mean, hist_sd = indicators(histFixed)
    if force_sd >= 0: hist_sd = force_sd

    # If the approximation y value at the x value of the mode is much to low,
    # the standard deviation of the approximation is to large. This will be
    # decreased step by step in this case.
    step = 1
    max_fraction = 0.95 if is_first_step else 1.1

    # Not all distributions are defined for x>=0. (The support of the
    # (gamma distribution is for example x>0). Therefore the x=0
    # histogram value has to be moved a little bit.
    x = np.arange(0, len(hist), 1, dtype=np.float64)
    x[0] = 0.0001

    sd = hist_sd
    while True:
        mean_est = estimate_mean(hist_mode, hist_mean, sd)
        if mean_est < 0.001: mean_est = 0.001
        dist = fit_distribution(mean_est, sd)

        est_pdf_atMode = dist.pdf(hist_mode)
        fraction = 1 / est_pdf_atMode * hist[hist_mode] if est_pdf_atMode > 0 else 1
        fraction = fraction / histFixed_sum
        if fraction < max_fraction or step >= 10: break
        sd = sd * 0.8
        step += 1

    pdf = np.vectorize(lambda x: dist.pdf(x))
    est_pdf = pdf(x)
    est_pdf = est_pdf * fraction * histFixed_sum
    residuum = hist - est_pdf

    # Remove edge effects (based on distortions in the measured values) from the residual
    first_index = 0
    while residuum.argmax() == first_index:
        residuum[first_index] = 0
        first_index += 1

    # Plot results
    if show_plot:
        plt.subplots(figsize=(16, 9))
        plt.plot(hist[0:250], color='blue', label='Messwerte')
        plt.plot(est_pdf[0:250], color='red', label='Näherung')
        plt.plot(residuum[0:250], color='gray', label='Residuum')
        plt.ylim(0, plt.ylim()[1])
        plt.legend()

    # Print text results
    if show_info:
        print("mean=", round(mean_est, 1), ", sd=", round(sd, 1), ", fraction=", round(fraction_all * fraction * 100), "%", sep="")

    parameters = {'mean': mean_est, 'sd': sd, 'mode': hist_mode, 'fraction': fraction}
    return parameters, residuum


def fraction_estimator(histogram: np.ndarray, distribution_type: callable, parameters1: list, parameters2: list) -> tuple[float, float]:
    """Optimizes the fractions of two distributions to match the histogram values as good as possible.

    Args:
        histogram (np.ndarray): Histogram values to be described by the distributions
        distribution_type (callable): Function for getting a frozen distribution object from given mean and standard deviation
        parameters1 (list): List containing "mode", "mean", "sd" and "fraction" of the first distribution
        parameters2 (list): List containing "mode", "mean" and "sd" of the second distribution

    Returns:
        tuple[float, float]: Optimized fractions of the two distributions
    """
    histogram_sum = sum(histogram)
    initial_fraction = parameters1['fraction']
    mode1 = parameters1['mode'] if parameters1['mode'] > 0 else 1
    mode2 = parameters2['mode'] if parameters2['mode'] > 0 else 1
    distribution1 = distribution_type(parameters1['mean'], parameters1['sd'])
    distribution2 = distribution_type(parameters2['mean'], parameters2['sd'])
    histogram_y1 = histogram[mode1] / histogram_sum
    histogram_y2 = histogram[mode2] / histogram_sum

    def f(fraction):
        y1 = distribution1.pdf(mode1) * fraction + distribution2.pdf(mode1) * (1 - fraction)
        y2 = distribution1.pdf(mode2) * fraction + distribution2.pdf(mode2) * (1 - fraction)
        delta = abs(y1 - histogram_y1) / histogram_y1 + abs(y2 - histogram_y2) / histogram_y2
        return delta

    result = minimize(f, x0=initial_fraction, bounds=[(0, 1)])
    return result.x[0], 1 - result.x[0]


def calc_combined_approximation(steps: list) -> np.ndarray:
    """Builds a approximation histogram of multiple partial histograms.

    Args:
        steps (list): List containing sub lists which contain "pdf" and "fraction" each.

    Returns:
        np.ndarray: Combined approximation histogram.
    """
    approximation = np.zeros(len(steps[0]['pdf']))
    fraction_used = 0

    for step in steps:
        fraction = (1 - fraction_used) * step["fraction"]
        fraction_used += fraction
        approximation += step["pdf"] * fraction

    return approximation
