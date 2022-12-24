"""Auxiliary functions for handling histogram and pdf values."""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


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


def indicators(histogram: list[float], show_info: bool = False) -> tuple[float, float, float]:
    """Calculates mode, mean and standard deviation from histogram values list.

    Args:
        histogram (list): List containing the histogram values
        show_info (bool, optional): If true, additionally prints the calculated values. Defaults to False.

    Returns:
        tuple[float]: Tuple of mode, mean and standard deviation
    """
    mode = histogram.argmax()
    x = np.arange(0, len(histogram))
    if max(histogram) == 0:
        mean = 0
        sd = 0
    else:
        mean = np.average(x, weights=histogram)
        sd = math.sqrt(max(0, np.average(x * x, weights=histogram) - mean**2))

    if show_info:
        print("mode=", mode, ", mean=", round(mean, 1), ", sd=", round(sd, 1), sep="")

    return mode, mean, sd


def build_pdf(steps: int, distribution_type: scipy.stats.rv_continuous, mean: float, sd: float) -> np.ndarray:
    """Generates pdf values for a given distribution.

    Args:
        steps (int): the pdf values will be in range 0 to steps-1
        distribution_type (scipy.stats.rv_continuous): Distribution type object
        mean (float): Mean of the distribution
        sd (float): Standard deviation of the distribution

    Returns:
        np.ndarray: Pdf values for the distribution
    """
    x = np.arange(0, steps, 1, dtype=np.float64)
    x[0] = 0.0001

    dist = distribution_type(mean, sd)
    pdf = np.vectorize(lambda x: dist.pdf(x))
    return pdf(x)


def plot_pdf(histogram: np.ndarray, approximation: np.ndarray, max_x: int) -> None:
    """Plots a distribution given as a histogram, an approximation for the distribution
    and the residuum between both.

    Args:
        histogram (np.ndarray): Distribution to be plotted
        approximation (np.ndarray): Approximation to be plotted
        max_x (int): Maximum x axis value of the plot
    """
    histogram_pdf = histogram / histogram.sum()
    approximation_pdf = approximation / approximation.sum()
    residuum_pdf = np.abs(histogram_pdf - approximation_pdf)

    plt.subplots(figsize=(16, 9))
    plt.plot(histogram_pdf[0:max_x], color='blue', label='Measured values (pdf)')
    plt.plot(approximation_pdf[0:max_x], color='red', label='Approximation (pdf)')
    plt.plot(residuum_pdf[0:max_x], color='gray', label='Residuum')
    plt.legend()
    plt.title("Measurement series and approximation")

    _, y_max = plt.ylim()
    y_max = min(y_max, 0.05)
    plt.ylim((0, y_max))
