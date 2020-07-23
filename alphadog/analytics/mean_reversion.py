"""
Functionality to test mean reversion and/or stationarity in time series.
"""
import pandas as pd

from statsmodels.tsa.stattools import adfuller


def augmented_dickey_fuller(df, significance_level=10):
    """
    Augmented Dickey-Fuller test for mean reversion.

    Parameters
    ----------
    df: pandas.DataFrame
        Time series to test.
    significance_level: {10, 5, 1}
        Significance level percentage to use. Defaults to 10%

    Returns
    -------
    significant: bool
        Whether the time series is significant at the given significance level.
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
    # TODO
    res = adfuller(df, maxlag=1, regression='c')
    test_stat = res[0]
    p_value = res[1]
    critical_stat = res[4][f"{significance_level}%"]
    ic_best = res[5]

    significant = test_stat <= critical_stat

    return significant, test_stat


def hurst_exponent(df):
    """
    Hurst exponent of a time series, indicating stationarity.

    Parameters
    ----------
    df: pandas.DataFrame
        Time series to test.

    Returns
    -------

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
    Ernest Chan (2013), Quantitative Trading. pp 41, 44-50.
    """
    return


def variance_test():
    """

    Returns
    -------

    Notes
    -----
    The variance test for the Hurst exponent tests whether:
    .. math:: var( z_t - z_{t-T} ) / T * var( z_t - z_{t-1} )
    is equal to 1

    References
    ----------
    Ernest Chan (2013), Quantitative Trading. pp 44-45.
    """
    pass


def mean_reversion_half_life(df):
    """

    Parameters
    ----------
    df

    Returns
    -------

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

    Thus, the price decays exponentially to the value -mu/lambda
    with a half-life of decay equal to -ln(2)/lambda

    The half-life can then be used for the lookback of the mean-reverting signal.

    References
    ----------
    Ernest Chan (2013), Quantitative Trading. pp 46-48.
    """
    pass


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
