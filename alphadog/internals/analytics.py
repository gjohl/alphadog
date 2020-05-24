"""
internals: any generic calculations
-- volatility
-- beta
-- correlation
-- covariance
-- ewma
-- kalman filter
-- state space models
-- ML libraries...
-- Sharpe ratio
-- drawdowns
-- autocorrelations
-- Granger causality
"""
import numpy as np
import pandas as pd


# TODO: add a robust vol calc that floors the vol at a rolling min value


def cross_sectional_mean(df, name='combined'):
    """
    Aggregate multiple columns by taking the cross-sectional mean.

    Parameters
    ----------
    df: pd.DataFrame

    name

    Returns
    -------

    """
    return df.mean(axis=1).to_frame(name)


def returns(price_df, return_type, percent=False):
    """
    Calculate the returns of a price series.

    Arithmetic (simple) returns aggregate across assets.
    Geometric (log) returns aggregate across time.

    Parameters
    ----------
    price_df: pd.DataFrame
        Price time series to calculate returns of.
    return_type: str
        Either 'arithmetic' or 'geometric'.
        Type of returns to calculate.
    percent: bool
        Whether to give the results as a percentage rather than a decimal.
        Default False.

    Returns
    -------
    returns_df: pd.DataFrame
        Time series of returns per column of the input DataFrame.
    """
    if return_type == 'arithmetic':
        returns_df = arithmetic_returns(price_df)
    elif return_type == 'geometric':
        returns_df = geometric_returns(price_df)
    else:
        raise NotImplementedError(f"return_type must be 'arithmetic' or 'geometric'. "
                                  f"Got {return_type}")

    if percent:
        returns_df *= 100.

    return returns_df


def arithmetic_returns(price_df):
    """
    Calculate the arithmetic returns of a price series.

    This is calculated as:
    r(t) = p(t) / p(t-1) - 1

    Parameters
    ----------
    price_df: pd.DataFrame
        Price time series to calculate returns of.

    Returns
    -------
    pd.DataFrame
        Time series of returns per column of the input DataFrame.
    """
    return price_df / price_df.shift() - 1


def geometric_returns(price_df):
    """
    Calculate the geometric returns of a price series.

    This is calculated as:
    r(t) = ln(p(t) / p(t-1)) = ln(p(t)) - ln(p(t-1))

    Parameters
    ----------
    price_df: pd.DataFrame
        Price time series to calculate returns of.

    Returns
    -------
    pd.DataFrame
        Time series of returns per column of the input DataFrame.
    """
    return np.log(price_df) - np.log(price_df.shift())
