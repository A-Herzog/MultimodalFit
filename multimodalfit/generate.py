"""Auxiliary functions for generating pseudo random numbers and histograms."""

import math
import random
import numpy as np
import time


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


def rng_lognormal(mean: float, sd: float) -> callable:
    """Generates a random numer callable for log-normal distributed pseudo random numbers

    Args:
        mean (float): Mean of the log-normal distribution
        sd (float): Standard deviation of the log-normal distribution

    Returns:
        callable: Lambda expression which will generate a pseudo random number on each call.
    """
    mu = math.log(mean**2 / math.sqrt(sd**2 + mean**2))
    sigma = math.sqrt(math.log((sd**2 / mean**2) + 1))
    return lambda: random.lognormvariate(mu, sigma)


def rng_gamma(mean: float, sd: float) -> callable:
    """Generates a random numer callable for gamma distributed pseudo random numbers

    Args:
        mean (float): Mean of the gamma distribution
        sd (float): Standard deviation of the gamma distribution

    Returns:
        callable: Lambda expression which will generate a pseudo random number on each call.
    """
    beta = mean / (sd * sd)
    alpha = mean * beta
    beta = 1 / beta
    return lambda: random.gammavariate(alpha, beta)


def histogram(mean: float, sd: float, n: int, max_x: int, rounding: callable = np.round, rng: callable = rng_lognormal) -> np.ndarray:
    """Generates a histogram based on pseudo random numbers of a given distribution

    Args:
        mean (float): Mean of the distribution
        sd (float): Standard deviation of the distribution
        n (int): Number of pseudo random numbers to be generated
        max_x (int): Maximum x value of the histogram
        rounding (callable, optional): Function to be used when mapping a random number of a histogram step. Defaults to np.round.
        rng (callable, optional): Pseudo random number generator type to be used. Defaults to rng_lognormal.

    Returns:
        np.ndarray: Histogram based on pseudo random numbers
    """
    generator = rng(mean, sd)
    values = rounding([generator() for _ in range(n)])
    values_limit = np.where(values < max_x, values, max_x - 1)
    hist, _ = np.histogram(values_limit, bins=max_x, range=(0, max_x - 1))
    return hist


def generate_bimodal(mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int, max_x: int, rounding: callable = np.round, rng: callable = rng_lognormal, show_info: bool = False) -> np.ndarray:
    """Generates a histogram based on pseudo random numbers of two given distributions

    Args:
        mean1 (float): Mean of the first distribution
        sd1 (float): Standard deviation of the first distribution
        n1 (int): Number of pseudo random numbers to be generated based on the first distribution
        mean2 (float): Mean of the second distribution
        sd2 (float): Standard deviation of the second distribution
        n2 (int): Number of pseudo random numbers to be generated based on the second distribution
        max_x (int): Maximum x value of the histogram
        rounding (callable, optional): Function to be used when mapping a random number of a histogram step. Defaults to np.round.
        rng (callable, optional): Pseudo random number generator type to be used. Defaults to rng_lognormal.
        show_info (bool, optional): If true, prints out how long it took to create the histogram. Defaults to False.

    Returns:
        np.ndarray: Histogram based on pseudo random numbers
    """
    start = time.time()
    histA = histogram(mean1, sd1, n1, max_x, rng=rng)
    histB = histogram(mean2, sd2, n2, max_x, rng=rng)
    result = histA + histB

    if show_info:
        print("Measurement series generated in", round(time.time() - start, 1), "seconds.")

    return result
