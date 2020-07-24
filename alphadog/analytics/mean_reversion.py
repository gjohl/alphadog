"""
Functionality to test mean reversion and/or stationarity in time series.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from statsmodels.tsa.stattools import adfuller


def augmented_dickey_fuller(df, significance_level=0.1):
    """
    Augmented Dickey-Fuller test for mean reversion.

    Parameters
    ----------
    df: pandas.DataFrame
        Time series to test.
    significance_level: float
        Significance level percentage to use. Defaults to 0.1 (10%).

    Returns
    -------
    significant: bool
        Whether the time series is significant at the given significance level.
    p_value: float
        The p-value of the test statistic.
    test_stat: float
        The test statistic.

    Notes
    -----
    If a time series is mean-reverting then future price moves are proportional to the
    difference between the current price and the mean price.
    I.e.
    .. math:: y_{t+1} - y_t = k * ( y_t - \bar{y} )

    The Augmented Dickey Fuller test checks whether we can reject the null hypothesis
    that the proportionality constant k is 0.

    The test statistic is k/SE(k).

    References
    ----------
    Ernest Chan (2013), Quantitative Trading. pp 41-44.
    """
    res = adfuller(df, maxlag=1, regression='c')
    p_value = res[1]
    test_stat = res[0]
    significant = p_value <= significance_level

    return significant, p_value, test_stat


def hurst_exponent(df):
    """
    Hurst exponent of a time series, indicating stationarity.

    Parameters
    ----------
    df: pandas.DataFrame
        Log price time series to test.

    Returns
    -------
    float
        The Hurst exponent for the given timeseries.

    Notes
    -----
    The time series is considered stationary if the variance of the log price increases
    sublinearly, i.e. slower than a geometric random walk.

    Consider prices y_t and log prices z_t = log(y_t)
    For a geometric random walk:
    .. math:: <|z(t+T) - z(t)|^2> ~ T
    where T is an arbitrary time lag.

    If the time series is stationary, the right side does not hold.
    Instead we replace T with T^(2H), where H is the Hurst exponent.

    H < 0.5 for a mean reverting series
    H = 0.5 for a geometric random walk
    H > 0.5 for a trending series

    References
    ----------
    [1] Ernest Chan (2013), Quantitative Trading. pp 41, 44-50.
    [2] https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/
    """
    # TODO: lags should be dynamic
    # TODO support multi-col timeseries? Maybe not worth it.
    lags = range(1, 100)
    ts_variance = [np.nanvar(df.diff(lag)) for lag in lags]
    linear_reg = np.polyfit(np.log(lags), np.log(ts_variance), 1)

    return linear_reg[0]


def variance_ratio_test(df, lag=2, significance_level=0.1):
    """
    Variance ratio test for the statistical significance of the Hurst exponent.

    Parameters
    ----------
    df: pandas.DataFrame
        Log price time series to test.
    lag: int
        Lag to check for significance. Defaults to 2, i.e. an AR(1) process.
    significance_level: float
        Significance level percentage to use. Defaults to 0.1 (10%).

    Returns
    -------
    significant: bool
        Whether the time series is significant at the given significance level.
    p_value: float
        The p-value of the test statistic.
    test_stat: float
        The test statistic.

    Notes
    -----
    The variance test for the Hurst exponent tests whether:
    .. math:: var( z_t - z_{t-T} ) / T * var( z_t - z_{t-1} )
    is equal to 1

    It is possible to standardise the statistic as:
    sqrt(2*n) * (sigma_b^2 / sigma_a^2 - 1) / sqrt(2)
    This follows a N(0,1) Normal distribution so Z-scores can be used to test the significance.
    See ref [3].

    References
    ----------
    [1] Ernest Chan (2013), Quantitative Trading. pp 44-45.
    [2] https://medium.com/bluekiri/simple-stationarity-tests-on-time-series-ad227e2e6d48
    [3] https://breakingdownfinance.com/finance-topics/finance-basics/variance-ratio-test/
    """
    variance_ratio = np.nanvar(df.diff(lag)) / (lag * np.nanvar(df.diff(1)))
    num_obs = df.shape[0]
    test_stat = np.sqrt(num_obs) * (variance_ratio - 1)
    p_value = norm.sf(np.abs(test_stat))
    significant = p_value <= significance_level

    return significant, p_value, test_stat


def mean_reversion_half_life(df):
    """
    The half-life of a mean-reverting Ornstein-Uhlbeck process.

    This is computed using a linear regression of return as the dependent variable
    against previous price as the independent variable which gives lambda as the slope and mu as the
    intercept.

    The half-life is then calculated as -ln(2)/lambda .

    Parameters
    ----------
    df: pandas.DataFrame
        Time series to test.

    Returns
    -------
    halflife: float
        The half-life of the mean-reverting process.

    Notes
    -----
    In discrete-form, the Augmented Dickey-Fuller test assumes price changes follow the model:
    .. math: \delta y_t = \lambda * y_{t-1} + \mu + E
    where E is Gaussian noise

    Performing a linear regression of ( y_t - y_{t-1} ) as the dependent variable
    against y_{t-1} as the independent variable gives lambda as the slope and mu as the
    intercept.

    In continuous-form, this is equivalent to prices following the mean-reverting
    Ornstein-Uhlenbeck process:
    .. math:: dy = (lambda * y_{t-1} + mu)dt + dE

    Rewriting in this way allows an analytical solution for the expectation of y_t:
    .. math:: y_bar = y_0 * exp(lambda * t) - mu / (lambda * (1 - exp(lambda *t)))

    If lambda is negative the process is mean-reverting.
    Thus, the price decays exponentially to the value -mu/lambda
    with a half-life of decay equal to -ln(2)/lambda .
    The half-life can then be used for the lookback of the mean-reverting signal.

    References
    ----------
    Ernest Chan (2013), Quantitative Trading. pp 46-48.
    """
    returns = df.diff().values.flatten()
    lagged_price = df.shift().values.flatten()

    # from statsmodels.regression.linear_model import OLS
    # linear_model = OLS(returns, lagged_price, missing='drop')
    # linear_model.fit()

    linear_reg = np.polyfit(returns[1:], lagged_price[1:], 1)
    slope = linear_reg[0]
    if slope >= 0:
        return np.nan
    halflife = -np.log(2) / slope

    return halflife


def mean_reversion_signal():
    """
    # TODO: move to signals once this is robust
    Opposite sign of momentum signal, with lookback span equal to the half-life of mean-reversion.

    Returns
    -------

    References
    ----------
    Ernest Chan (2013), Quantitative Trading. pp 49.
    """
    pass


def cointegrated_augmented_dickey_fuller():
    """
    Determine the stationarity of a combination of two time series.

    This also determines the hedge ratio, i.e. what proportion to combine the two securities in
    to create a stationary signal. This is useful for pairs trading of spreads.

    Returns
    -------

    References
    ----------
    Ernest Chan (2013), Quantitative Trading. pp 51-54.
    """
    pass


def johansen_test():
    """
    Determine the stationarity of a combination of an arbitrary number of time series.

    Returns
    -------

    References
    ----------
    Ernest Chan (2013), Quantitative Trading. pp 54-58.
    """
    pass
