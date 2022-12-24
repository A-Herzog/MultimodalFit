# MultimodalFit

MultimodalFit is a Python package for fitting a combination of multiple distributions to one measurement series. This allows to find out on which individual distributions the measured combined distribution consists.



## Installation

A PyPI package is not yet available. Just download and extract the zip package offered here.

The zip package also includes an example Jupyter notebook.



## Requirements

* **Python 3.9 or higher** is needed to execute MultimodalFit.
* **`numpy`**, **`matplotlib`** and **`scipy`** are used.



## Usage

The extraction of distributions from a histogram is done step by step. In each estimation step the distribution with the highest value at the mode is extracted. A single estimation step is done by calling `estimation_step`:
```
parameters, residuum = estimation_step(histogram, estimate_lognormal_mean, dist_lognormal)
```
The histogram input parameter is a `numpy.ndarray` containing the measured values (or the residuum from the previous step). The second parameter is a function for estimating the mean of a distribution from its mode and its standard deviation. The provided function `estimate_lognormal_mean` is used for the log-normal distribution. There is also a function `estimate_gamma_mean` which does the same for the gamma distribution. The Third parameter is a function that generates a `scipy` frozen distribution from a given mean and a given standard deviation. (`dist_lognormal` for the log-normal distribution and `dist_gamma` for the gamma distribution.)



## Example

```
# Generate a histogram containing values from the combined distributions
histogram = generate_bimodal(mean1, sd1, n1, mean2, sd2, n2, max_x, rng=rng_lognormal, show_info=True)

# Estimate parameters of the first contained distribution
parameters1, residuum1 = estimation_step(histogram, estimate_lognormal_mean, dist_lognormal)

# Estimate parameters of the second contained distribution
parameters2, residuum2 = estimation_step(residuum1, estimate_lognormal_mean, dist_lognormal)

# Optimize which fractions the two distributions have in the summed distribution
fraction1, _ = fraction_estimator(histogram, dist_lognormal, parameters1, parameters2)
```

If the estimation works well, `parameters1["mean"]`and `parameters1["sd"]` should be close to `mean1` and `sd1`. And `parameters2["mean"]`and `parameters2["sd"]` should be close to `mean2` and `sd2`.

See [`example_estimator.ipynb`](example_estimator.ipynb) for a complete example.



## Warteschlangensimulator

The algorithms tested here are also implemented in **[Warteschlangensimulator](https://github.com/A-Herzog/Warteschlangensimulator)**, which is an open source desktop application for modelling and simulation of queueing models. The multimodal distribution fitter is used as part of the input analysis tools.



# Contact

**Alexander Herzog**<br>
[TU Clausthal](https://www.tu-clausthal.de)<br>
[Simulation Science Center Clausthal / Göttingen](https://www.simzentrum.de/)<br>
alexander.herzog@tu-clausthal.de
