"""
Functionality to test mean reversion and/or stationarity in time series.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from alphadog.internals.exceptions import InputDataError, ParameterError


ALLOWED_JOHANSEN_SIGNIFICANCE = {0.1, 0.05, 0.01}


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
    test_stat = res[0]
    p_value = res[1]
    significant = p_value <= significance_level

    return significant, p_value, test_stat


def hurst_exponent(df, min_lags=100):
    """
    Hurst exponent of a time series, indicating stationarity.

    Parameters
    ----------
    df: pandas.DataFrame
        Log price time series to test.
    min_lags: int
        The minimum number of lags to use to calculate the Hurst exponent.
        If the given time series is shorter than this, raise an error.

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
    # Get the number of lags to use
    num_obs = df.shape[0]
    max_lags = max(min_lags, num_obs // 4)
    if num_obs <= min_lags:
        raise InputDataError(f"Number of observations ({num_obs}) is less than the required"
                             f"minimum number of lags ({min_lags})")

    # Linear regression of variance against lag
    lags = range(1, max_lags)
    ts_variance = [np.nanvar(df.diff(lag)) for lag in lags]
    linear_model = sm.OLS(np.log(ts_variance), np.log(lags))
    results = linear_model.fit()
    slope = results.params[0]

    return slope * 0.5


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
    returns = df.diff()
    lagged_price = df.shift()
    lagged_price = sm.add_constant(lagged_price)

    linear_model = sm.OLS(returns, lagged_price, missing='drop')
    results = linear_model.fit()
    coefficients = results.params.values
    slope = coefficients[1]

    if slope >= 0:
        return np.nan
    halflife = -np.log(2) / slope

    return halflife


def cointegrated_augmented_dickey_fuller(df1, df2, significance_level=0.1):
    """
    Determine the stationarity of a combination of two time series.

    This also determines the hedge ratio, i.e. what proportion to combine the two securities in
    to create a stationary signal. This is useful for pairs trading of spreads.

    df1: pd.DataFrame
        One of two time series to test. This will typically be log prices.
    df2: pd.DataFrame
        One of two time series to test. This will typically be log prices.
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
    The steps for a cointegrated ADF test are essentially:
    1. Perform a linear regression of the two time series to find the optimal hedge ratio.
    2. Create a portfolio of the two timeseries
    3. Perform an ADF test on the portfolio.

    The results are order-dependent and the hedge ratio is typically quite volatile.

    References
    ----------
    [1] Ernest Chan (2013), Quantitative Trading. pp 51-54.
    [2] https://medium.com/bluekiri/cointegration-tests-on-time-series-88702ea9c492
    """
    comb_df = pd.concat([df1, df2], axis=1).dropna()
    res = coint(comb_df.iloc[:, 0], comb_df.iloc[:, 1], trend='c')
    test_stat = res[0]
    p_value = res[1]
    significant = p_value <= significance_level

    return significant, p_value, test_stat


def johansen_test(combined_df, significance_level=0.1, method='eigen'):
    """
    Determine the stationarity of a combination of an arbitrary number of time series.

    Also gives the eignevector which is the optimal hedge ratio for the largest eigenvalue
    which represents the "strongest" mean reversion.

    Parameters
    ----------
    combined_df: NxM pd.DataFrame
        Combined DataFrame of all time series we wish to test for stationarity.
    significance_level: {0.1, 0.05, 0.01}
        Significance level percentage to use. Defaults to 0.1 (10%).
    method: {'eigen', 'trace}
        Method used to determine significance. Defaults to 'eigen'.

    Returns
    -------
    significant: bool
        Whether the time series is significant at the given significance level.
    test_stat: float
        The test statistic.
    eigenvalue: float
        The eigenvalue of the "strongest" combination.
    eigenvector: 1xM np.array
        The eignevector representing the optimal hedge ratio between the given time series.

    Examples
    --------
    significant, test_stat, eigenvalue, eigenvector = johansen_test(combined_df)
    stationary_df = comb_df.iloc[:, 0] * eigenvector[0] + comb_df.iloc[:, 1] * eigenvector[1]


    Notes
    -----
    Unlike the cointegrated augmented Dickey-Fuller test, the Johansen test can be applied
    to an arbitrary number of input signals and is not sensitive to the ordering.

    The rank of the matrix of input signals gives the number of independent portfolios that can
    be formed. This is calculated in two ways: using the trace statistics and using the eigenvalues.

    We choose the eigenvalues/eigenvectors corresponding to the largest eigenvalue as this
    represents the "strongest" mean reversion.

    References
    ----------
    [1] Ernest Chan (2013), Quantitative Trading. pp 54-58.
    [2] https://medium.com/bluekiri/cointegration-tests-on-time-series-88702ea9c492
    [3] https://blog.quantinsti.com/johansen-test-cointegration-building-stationary-portfolio/
    """
    if significance_level not in ALLOWED_JOHANSEN_SIGNIFICANCE:
        raise ParameterError(f"significance_level must be in {ALLOWED_JOHANSEN_SIGNIFICANCE}. "
                             f"Got {significance_level}")

    # Johansen test with constant term but no linear trend, 1 lagged value allowed.
    result = coint_johansen(combined_df, det_order=0, k_ar_diff=1)

    # Test for significance
    if method == 'eigen':
        critical_array = result.max_eig_stat_crit_vals
        test_stat = result.max_eig_stat[0]
    elif method == 'trace':
        critical_array = result.trace_stat_crit_vals
        test_stat = result.trace_stat[0]
    else:
        raise ParameterError(f"method must be 'eigen' or 'trace'. Got {method}")

    sig_level_to_index_map = {0.1: 0, 0.05: 1, 0.01: 2}
    critical_index = sig_level_to_index_map[significance_level]
    critical_value = critical_array[0, critical_index]
    significant = test_stat >= critical_value

    # Get the optimal eigenvalue and eigenvector
    eigenvalue = result.eig[0]
    eigenvector = result.evec[:, 0]

    return significant, test_stat, eigenvalue, eigenvector


def mean_reversion_signal():
    """
    Mean reversion signal.

    # TODO: move to signals once this is robust
    Opposite sign of momentum signal, with lookback span equal to the half-life of mean-reversion.

    Returns
    -------

    References
    ----------
    Ernest Chan (2013), Quantitative Trading. pp 49.
    """
    pass
